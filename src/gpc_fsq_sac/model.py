"""Exact ProtoMotions FSQ bottleneck adapted to the released SAC actor contract."""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path

import torch
from rsl_rl.models import SACCriticModel
from rsl_rl.modules import EmpiricalNormalization, MLP
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal

from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.common.fsq import FiniteScalarQuantizer
from protomotions.agents.utils.normalization import RunningMeanStd

from .config import FSQSACModelConfig

log = logging.getLogger(__name__)


class TrustRegionSACCritic(SACCriticModel):
    """Apply a slowly moving behavior-policy anchor only to SAC actor gradients."""

    def __init__(
        self,
        *args,
        actor_reference: nn.Module,
        actor_trust_region_coef: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.actor_trust_region_coef = actor_trust_region_coef
        self.actor_reference = copy.deepcopy(actor_reference)
        self.actor_reference.requires_grad_(False)
        self.actor_reference.eval()

    def evaluate_all_q(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1, q2 = super().evaluate_all_q(obs, actions)
        if self.actor_trust_region_coef <= 0 or not actions.requires_grad:
            return q1, q2
        with torch.no_grad():
            self.actor_reference.eval()
            reference_actions = self.actor_reference(obs.detach().clone())
        trust_penalty = (actions - reference_actions).square().mean(
            dim=-1,
            keepdim=True,
        )
        penalty = self.actor_trust_region_coef * trust_penalty
        return q1 - penalty, q2 - penalty

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
        return (
            actor(obs.detach().clone()) - self.actor_reference(obs.detach().clone())
        ).square().mean()


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
        self.ppo_compatible_normalization = self._as_bool(
            config.ppo_compatible_normalization
        )
        self.freeze_normalization = self._as_bool(config.freeze_normalization)
        state_dim = obs["max_coords_obs"].shape[-1]
        target_dim = obs["mimic_target_poses"].shape[-1]
        self.output_dim = action_dim
        self.normalization_clip = config.normalization_clip
        if self.ppo_compatible_normalization:
            self.state_normalizer = nn.Identity()
            self.target_normalizer = RunningMeanStd(
                fabric=None,
                shape=(target_dim,),
                device="cpu",
                clamp_value=config.normalization_clip,
            )
            self.decoder_normalizer = RunningMeanStd(
                fabric=None,
                shape=(state_dim + config.num_fsq_scalars,),
                device="cpu",
                clamp_value=config.normalization_clip,
            )
        else:
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
            self.decoder_normalizer = nn.Identity()
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
        nn.init.normal_(last_linear.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(last_linear.bias)
        Normal.set_default_validate_args(False)

    def _features(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        raw_state = obs["max_coords_obs"]
        raw_target = obs["mimic_target_poses"]
        state = self.state_normalizer(raw_state)
        if self.ppo_compatible_normalization:
            target = self.target_normalizer.normalize(raw_target)
        else:
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
        if self.ppo_compatible_normalization:
            decoder_input = self.decoder_normalizer.normalize(decoder_input)
        mean = self.decoder(decoder_input)
        return mean, codes

    def _update_distribution(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        mean, codes = self._features(obs)
        std = self.effective_log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)
        return mean, codes

    def _squash_and_scale(self, value: torch.Tensor) -> torch.Tensor:
        return self.action_range * torch.tanh(value) + self.action_bias

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
        return self._squash_and_scale(value)

    def sample_action_logp(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_distribution(obs)
        value = self.distribution.rsample()
        squashed = torch.tanh(value)
        action = self.action_range * squashed + self.action_bias
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

    def update_normalization(self, obs: TensorDict) -> None:
        if self.freeze_normalization:
            return
        if self.ppo_compatible_normalization:
            self.target_normalizer.record_moments(obs["mimic_target_poses"])
            with torch.no_grad():
                target = self.target_normalizer.normalize(
                    obs["mimic_target_poses"]
                )
                continuous_latent = self.encoder(target)
                codes = (
                    self.quantizer.quantize(continuous_latent)
                    if self.use_fsq
                    else continuous_latent
                )
                decoder_input = torch.cat(
                    [obs["max_coords_obs"], codes],
                    dim=-1,
                )
            self.decoder_normalizer.record_moments(decoder_input)
        elif hasattr(self.state_normalizer, "update"):
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
            action = torch.tanh(mean)
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
                "policy/pre_tanh_entropy": entropy.mean(),
            }

    def load_ppo_actor_checkpoint(self, checkpoint_path: str) -> None:
        if not self.ppo_compatible_normalization or not self.use_fsq:
            raise ValueError(
                "PPO actor warm-start requires FSQ and PPO-compatible normalization"
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint["model"]
        with torch.no_grad():
            for module_name, module in (
                ("encoder", self.encoder),
                ("decoder", self.decoder),
            ):
                source_prefix = f"_actor.mu.{module_name}.mlp."
                for key, value in state.items():
                    if key.startswith(source_prefix):
                        destination = key.removeprefix(source_prefix)
                        module.state_dict()[destination].copy_(value)

            self.log_std.copy_(state["_actor.logstd"])
            for module_name, source_name in (
                ("target_normalizer", "_actor.mu.encoder.norm.running_obs_norm"),
                ("decoder_normalizer", "_actor.mu.decoder.norm.running_obs_norm"),
            ):
                normalizer = getattr(self, module_name)
                normalizer.mean.copy_(state[f"{source_name}.mean"])
                normalizer.var.copy_(state[f"{source_name}.var"])
                normalizer.count.copy_(state[f"{source_name}.count"])


class FSQSACModel(BaseModel):
    """ProtoMotions evaluator/checkpoint wrapper around SAC actor and twin critic."""

    config: FSQSACModelConfig

    def __init__(self, config: FSQSACModelConfig, obs: TensorDict, action_dim: int):
        super().__init__(config)
        self.actor = FSQSACActor(obs, action_dim, config)
        if config.ppo_actor_checkpoint is not None:
            checkpoint_path = Path(config.ppo_actor_checkpoint)
            if checkpoint_path.is_file():
                self.actor.load_ppo_actor_checkpoint(str(checkpoint_path))
            else:
                log.warning(
                    "PPO warm-start source %s is unavailable; constructing the "
                    "actor for self-contained checkpoint loading.",
                    checkpoint_path,
                )
        self.critic = TrustRegionSACCritic(
            obs=obs,
            obs_groups={
                "critic": ["max_coords_obs", "mimic_target_poses"],
            },
            obs_set="critic",
            output_dim=1,
            hidden_dims=list(config.critic_hidden_dims),
            activation="relu",
            obs_normalization=config.normalize_observations,
            num_actions=action_dim,
            actor_reference=self.actor,
            actor_trust_region_coef=config.actor_trust_region_coef,
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
