"""Exact ProtoMotions FSQ bottleneck adapted to the released SAC actor contract."""

from __future__ import annotations

import copy
import math

import torch
from rsl_rl.models import SACCriticModel
from rsl_rl.modules import EmpiricalNormalization, MLP
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal

from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.common.fsq import FiniteScalarQuantizer
from .config import FSQSACModelConfig

class TrustRegionSACCritic(SACCriticModel):
    """Apply SAC's own slowly moving behavior anchor only to actor gradients.

    The reference is always copied from the randomly initialized SAC actor in
    this process. There is no checkpoint-loading or teacher-policy path.
    """

    def __init__(
        self,
        *args,
        actor_reference: nn.Module,
        actor_trust_region_coef: float,
        actor_reference_motion_ids: list[int] | None,
        reference_residual_actions: bool,
        reference_action_gain: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.actor_trust_region_coef = actor_trust_region_coef
        self.actor_reference_motion_ids = tuple(actor_reference_motion_ids or ())
        self.reference_residual_actions = reference_residual_actions
        self.reference_action_gain = reference_action_gain
        self.num_motion_heads = int(kwargs.get("output_dim", 1))
        if self.num_motion_heads < 1:
            raise ValueError("critic output_dim must be positive")
        self.actor_reference = copy.deepcopy(actor_reference)
        self.actor_reference.requires_grad_(False)
        self.actor_reference.eval()
        self.register_buffer(
            "critic_action_bias",
            torch.zeros(actor_reference.output_dim),
        )
        self.register_buffer(
            "critic_action_range",
            torch.ones(actor_reference.output_dim),
        )

    @torch.no_grad()
    def set_action_scaling(
        self,
        action_bias: torch.Tensor,
        action_range: torch.Tensor,
    ) -> None:
        """Keep environment scaling out of the critic feature magnitude.

        The actor emits calibrated environment actions. The released critic
        concatenates actions with normalized observations, so feeding narrow
        physical action ranges directly makes the critic effectively
        action-blind. Map actions back to the actor's canonical ``[-1, 1]``
        coordinates before every online and target Q evaluation.
        """
        self.critic_action_bias.copy_(action_bias)
        self.critic_action_range.copy_(action_range)
        self.actor_reference.action_bias.copy_(action_bias)
        self.actor_reference.action_range.copy_(action_range)
        self.actor_reference.log_action_range.copy_(
            torch.log(action_range).sum()
        )

    def action_center(self, obs: TensorDict) -> torch.Tensor:
        if self.reference_residual_actions:
            if "sac_reference_action" not in obs.keys():
                raise KeyError(
                    "sac_reference_action is required for residual actions"
                )
            return self.critic_action_bias + self.reference_action_gain * (
                obs["sac_reference_action"] - self.critic_action_bias
            )
        return self.critic_action_bias

    def normalize_actions(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return (actions - self.action_center(obs)) / self.critic_action_range

    def actor_reference_mask(self, obs: TensorDict) -> torch.Tensor:
        """Select replay rows whose behavior should retain the SAC anchor."""
        batch_size = obs.batch_size[0]
        if not self.actor_reference_motion_ids:
            return torch.ones(
                batch_size,
                1,
                device=self.critic_action_bias.device,
            )
        if "sac_motion_id" not in obs.keys():
            raise KeyError(
                "sac_motion_id is required when actor_reference_motion_ids is set"
            )
        motion_ids = obs["sac_motion_id"].reshape(batch_size, -1)[:, 0].long()
        reference_ids = torch.tensor(
            self.actor_reference_motion_ids,
            device=motion_ids.device,
            dtype=motion_ids.dtype,
        )
        return (motion_ids[:, None] == reference_ids[None, :]).any(
            dim=1,
            keepdim=True,
        ).to(dtype=self.critic_action_bias.dtype)

    def select_motion_q(
        self,
        obs: TensorDict,
        q_values: torch.Tensor,
    ) -> torch.Tensor:
        """Select the Q head for each replay row without exposing IDs to actor."""
        if self.num_motion_heads == 1:
            return q_values
        if "sac_motion_id" not in obs.keys():
            raise KeyError(
                "sac_motion_id is required when critic_num_motion_heads > 1"
            )
        motion_ids = obs["sac_motion_id"].reshape(q_values.shape[0], -1)[
            :, :1
        ].long()
        if torch.any(motion_ids < 0) or torch.any(
            motion_ids >= self.num_motion_heads
        ):
            raise IndexError(
                "sac_motion_id is outside the configured critic head range"
            )
        return q_values.gather(dim=-1, index=motion_ids)

    def forward(
        self,
        obs: TensorDict,
        masks=None,
        hidden_state=None,
        stochastic_output: bool = False,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if actions is None:
            raise ValueError("SAC critic requires actions")
        q_values = super().forward(
            obs,
            masks=masks,
            hidden_state=hidden_state,
            stochastic_output=stochastic_output,
            actions=self.normalize_actions(obs, actions),
        )
        return self.select_motion_q(obs, q_values)

    def evaluate_all_q(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1_all, q2_all = super().evaluate_all_q(
            obs,
            self.normalize_actions(obs, actions),
        )
        q1 = self.select_motion_q(obs, q1_all)
        q2 = self.select_motion_q(obs, q2_all)
        if self.actor_trust_region_coef <= 0 or not actions.requires_grad:
            return q1, q2
        with torch.no_grad():
            self.actor_reference.eval()
            reference_actions = self.actor_reference(obs.detach().clone())
        trust_penalty = (
            self.normalize_actions(obs, actions)
            - self.normalize_actions(obs, reference_actions)
        ).square().mean(dim=-1, keepdim=True)
        trust_penalty *= self.actor_reference_mask(obs)
        penalty = self.actor_trust_region_coef * trust_penalty
        return q1 - penalty, q2 - penalty

    def evaluate_all_target_q(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1_all, q2_all = super().evaluate_all_target_q(
            obs,
            self.normalize_actions(obs, actions),
        )
        return (
            self.select_motion_q(obs, q1_all),
            self.select_motion_q(obs, q2_all),
        )

    @torch.no_grad()
    def update_actor_reference(self, actor: nn.Module, tau: float) -> None:
        for reference, current in zip(
            self.actor_reference.parameters(),
            actor.parameters(),
            strict=True,
        ):
            reference.lerp_(current, tau)
        for reference, current in zip(
            self.actor_reference.buffers(),
            actor.buffers(),
            strict=True,
        ):
            if reference.is_floating_point():
                reference.lerp_(current, tau)
            else:
                reference.copy_(current)

    @torch.no_grad()
    def actor_reference_mse(
        self,
        actor: nn.Module,
        obs: TensorDict,
    ) -> torch.Tensor:
        self.actor_reference.eval()
        squared_error = (
            actor(obs.detach().clone()) - self.actor_reference(obs.detach().clone())
        ).square().mean(dim=-1, keepdim=True)
        mask = self.actor_reference_mask(obs)
        return (squared_error * mask).sum() / mask.sum().clamp(min=1)


class FSQSACActor(nn.Module):
    """Target encoder -> 40-scalar FSQ -> proprioceptive action decoder."""

    is_recurrent = False

    @staticmethod
    def _as_bool(value: bool | str) -> bool:
        return value.lower() == "true" if isinstance(value, str) else value

    def __init__(self, obs: TensorDict, action_dim: int, config: FSQSACModelConfig):
        super().__init__()
        learn_std = self._as_bool(config.learn_std)
        self.use_fsq = self._as_bool(config.use_fsq)
        self.freeze_normalization = self._as_bool(config.freeze_normalization)
        self.reference_residual_actions = self._as_bool(
            config.reference_residual_actions
        )
        self.reference_action_gain = config.reference_action_gain
        state_dim = obs["max_coords_obs"].shape[-1]
        target_dim = obs["mimic_target_poses"].shape[-1]
        self.output_dim = action_dim
        self.normalization_clip = config.normalization_clip
        self.state_normalizer = (
            EmpiricalNormalization(state_dim)
            if config.normalize_observations
            else nn.Identity()
        )
        self.target_normalizer = (
            EmpiricalNormalization(target_dim)
            if config.normalize_observations
            else nn.Identity()
        )
        self.encoder = MLP(
            target_dim,
            config.num_fsq_scalars,
            list(config.encoder_hidden_dims),
            "relu",
        )
        self.quantizer = FiniteScalarQuantizer(
            config.num_fsq_levels,
            config.num_fsq_scalars,
        )
        self.decoder = MLP(
            state_dim + config.num_fsq_scalars,
            action_dim,
            list(config.decoder_hidden_dims),
            "relu",
        )
        self.log_std = nn.Parameter(
            torch.full((action_dim,), math.log(config.initial_std)),
            requires_grad=learn_std,
        )
        self.min_log_std = config.min_log_std
        self.max_log_std = config.max_log_std
        self.register_buffer("action_bias", torch.zeros(action_dim))
        self.register_buffer("action_range", torch.ones(action_dim))
        self.register_buffer("log_action_range", torch.zeros(1))
        self.distribution: Normal | None = None

        last_linear = next(
            module for module in reversed(self.decoder) if isinstance(module, nn.Linear)
        )
        if self.reference_residual_actions:
            nn.init.zeros_(last_linear.weight)
        else:
            nn.init.normal_(last_linear.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(last_linear.bias)
        Normal.set_default_validate_args(False)

    def _features(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        raw_state = obs["max_coords_obs"]
        raw_target = obs["mimic_target_poses"]
        state = self.state_normalizer(raw_state)
        target = self.target_normalizer(raw_target)
        state = state.clamp(-self.normalization_clip, self.normalization_clip)
        target = target.clamp(-self.normalization_clip, self.normalization_clip)
        continuous_latent = self.encoder(target)
        codes = (
            self.quantizer.quantize(continuous_latent)
            if self.use_fsq
            else continuous_latent
        )
        decoder_input = torch.cat([state, codes], dim=-1)
        mean = self.decoder(decoder_input)
        return mean, codes

    def _update_distribution(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        mean, codes = self._features(obs)
        std = self.effective_log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)
        return mean, codes

    def action_center(self, obs: TensorDict) -> torch.Tensor:
        if self.reference_residual_actions:
            if "sac_reference_action" not in obs.keys():
                raise KeyError(
                    "sac_reference_action is required for residual actions"
                )
            return self.action_bias + self.reference_action_gain * (
                obs["sac_reference_action"] - self.action_bias
            )
        return self.action_bias

    def _squash_and_scale(
        self,
        value: torch.Tensor,
        obs: TensorDict,
    ) -> torch.Tensor:
        return self.action_range * torch.tanh(value) + self.action_center(obs)

    def forward(
        self,
        obs: TensorDict,
        masks=None,
        hidden_state=None,
        stochastic_output: bool = False,
        actions=None,
    ) -> torch.Tensor:
        del masks, hidden_state, actions
        mean, _ = self._update_distribution(obs)
        value = self.distribution.rsample() if stochastic_output else mean
        return self._squash_and_scale(value, obs)

    def sample_action_logp(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_distribution(obs)
        value = self.distribution.rsample()
        squashed = torch.tanh(value)
        action = self.action_range * squashed + self.action_center(obs)
        log_prob = self.distribution.log_prob(value).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - squashed.square() + 1e-6).sum(dim=-1, keepdim=True)
        log_prob -= self.log_action_range
        return action, log_prob

    @property
    def output_std(self) -> torch.Tensor:
        return self.effective_log_std.exp()

    @property
    def effective_log_std(self) -> torch.Tensor:
        return self.log_std.clamp(self.min_log_std, self.max_log_std)

    @torch.no_grad()
    def set_physical_action_std(self, physical_action_std: float) -> None:
        """Set vector exploration noise to a uniform physical-action scale."""
        if physical_action_std <= 0:
            raise ValueError("physical_action_std must be positive")
        canonical_std = physical_action_std / self.action_range
        canonical_std.clamp_(
            min=math.exp(self.min_log_std),
            max=math.exp(self.max_log_std),
        )
        self.log_std.copy_(canonical_std.log())

    def update_normalization(self, obs: TensorDict) -> None:
        if self.freeze_normalization:
            return
        if hasattr(self.state_normalizer, "update"):
            self.state_normalizer.update(obs["max_coords_obs"])
            self.target_normalizer.update(obs["mimic_target_poses"])

    def reset(self, dones=None, hidden_state=None) -> None:
        del dones, hidden_state

    def fsq_diagnostics(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            _, codes = self._features(obs)
            if not self.use_fsq:
                return {
                    "fsq/enabled": torch.tensor(0.0, device=codes.device),
                    "fsq/perplexity": torch.tensor(0.0, device=codes.device),
                    "fsq/code_abs_mean": codes.abs().mean(),
                    "fsq/code_saturation": torch.tensor(
                        0.0,
                        device=codes.device,
                    ),
                    "fsq/continuous_std": codes.std(),
                }
            return {
                "fsq/enabled": torch.tensor(1.0, device=codes.device),
                "fsq/perplexity": self.quantizer.calculate_perplexity(codes),
                "fsq/code_abs_mean": codes.abs().mean(),
                "fsq/code_saturation": (
                    codes.abs() >= self.quantizer.half_width.max()
                ).float().mean(),
            }

    def policy_diagnostics(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            mean, _ = self._features(obs)
            std = self.output_std
            action = self._squash_and_scale(mean, obs)
            shuffled_obs = obs.detach().clone()
            shuffled_obs["mimic_target_poses"] = torch.roll(
                shuffled_obs["mimic_target_poses"],
                shifts=1,
                dims=0,
            )
            shuffled_mean, _ = self._features(shuffled_obs)
            shuffled_action = self._squash_and_scale(shuffled_mean, shuffled_obs)
            relative_action = (
                action - self.action_center(obs)
            ) / self.action_range
            entropy = Normal(mean, std.expand_as(mean)).entropy().sum(dim=-1)
            return {
                "policy/std_min": std.min(),
                "policy/std_mean": std.mean(),
                "policy/std_max": std.max(),
                "policy/log_std_min": self.effective_log_std.min(),
                "policy/log_std_mean": self.effective_log_std.mean(),
                "policy/log_std_max": self.effective_log_std.max(),
                "policy/mean_abs": mean.abs().mean(),
                "policy/action_abs_mean": action.abs().mean(),
                "policy/action_std": action.std(),
                "policy/action_saturation": (action.abs() > 0.95).float().mean(),
                "policy/action_relative_saturation": (
                    relative_action.abs() > 0.95
                ).float().mean(),
                "policy/action_range_min": self.action_range.min(),
                "policy/action_range_mean": self.action_range.mean(),
                "policy/action_range_max": self.action_range.max(),
                "policy/action_bias_abs_mean": self.action_bias.abs().mean(),
                "policy/pre_tanh_entropy": entropy.mean(),
                "policy/target_action_sensitivity": (
                    action - shuffled_action
                ).abs().mean(),
            }

class FSQSACModel(BaseModel):
    """ProtoMotions evaluator/checkpoint wrapper around SAC actor and twin critic."""

    config: FSQSACModelConfig

    def __init__(self, config: FSQSACModelConfig, obs: TensorDict, action_dim: int):
        super().__init__(config)
        self.actor = FSQSACActor(obs, action_dim, config)
        self.critic = TrustRegionSACCritic(
            obs=obs,
            obs_groups={
                "critic": ["max_coords_obs", "mimic_target_poses"],
            },
            obs_set="critic",
            output_dim=config.critic_num_motion_heads,
            hidden_dims=list(config.critic_hidden_dims),
            activation="relu",
            obs_normalization=config.normalize_observations,
            num_actions=action_dim,
            actor_reference=self.actor,
            actor_trust_region_coef=config.actor_trust_region_coef,
            actor_reference_motion_ids=config.actor_reference_motion_ids,
            reference_residual_actions=config.reference_residual_actions,
            reference_action_gain=config.reference_action_gain,
        )
        if config.reference_residual_actions:
            if config.reference_residual_action_scale <= 0:
                raise ValueError(
                    "reference_residual_action_scale must be positive"
                )
            if not 0.0 <= config.reference_action_gain <= 1.0:
                raise ValueError("reference_action_gain must be in [0, 1]")
            with torch.no_grad():
                self.actor.action_bias.zero_()
                self.actor.action_range.fill_(
                    config.reference_residual_action_scale
                )
                self.actor.log_action_range.copy_(
                    torch.log(self.actor.action_range).sum()
                )
        self.critic.set_action_scaling(
            self.actor.action_bias,
            self.actor.action_range,
        )
        self.in_keys = list(config.in_keys)
        self.out_keys = list(config.out_keys)

    def forward(self, tensordict: TensorDict, log_internals: bool = False) -> TensorDict:
        action = self.actor(tensordict, stochastic_output=False)
        tensordict["action"] = action
        tensordict["mean_action"] = action
        if log_internals:
            for key, value in self.actor.fsq_diagnostics(tensordict).items():
                tensordict[key] = value.expand(tensordict.batch_size).clone()
        return tensordict
