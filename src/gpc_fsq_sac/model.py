"""Exact ProtoMotions FSQ bottleneck adapted to the released SAC actor contract."""

from __future__ import annotations

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


class FSQSACActor(nn.Module):
    """Target encoder -> 40-scalar FSQ -> proprioceptive action decoder."""

    is_recurrent = False

    def __init__(self, obs: TensorDict, action_dim: int, config: FSQSACModelConfig):
        super().__init__()
        state_dim = obs["max_coords_obs"].shape[-1]
        target_dim = obs["mimic_target_poses"].shape[-1]
        self.output_dim = action_dim
        self.normalization_clip = config.normalization_clip
        self.state_normalizer = (
            EmpiricalNormalization(state_dim) if config.normalize_observations else nn.Identity()
        )
        self.target_normalizer = (
            EmpiricalNormalization(target_dim) if config.normalize_observations else nn.Identity()
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
            torch.full((action_dim,), math.log(config.initial_std))
        )
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
        state = self.state_normalizer(obs["max_coords_obs"])
        target = self.target_normalizer(obs["mimic_target_poses"])
        state = state.clamp(-self.normalization_clip, self.normalization_clip)
        target = target.clamp(-self.normalization_clip, self.normalization_clip)
        continuous_latent = self.encoder(target)
        codes = self.quantizer.quantize(continuous_latent)
        mean = self.decoder(torch.cat([state, codes], dim=-1))
        return mean, codes

    def _update_distribution(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        mean, codes = self._features(obs)
        std = self.log_std.exp().expand_as(mean)
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
        return self.log_std.exp()

    def update_normalization(self, obs: TensorDict) -> None:
        if hasattr(self.state_normalizer, "update"):
            self.state_normalizer.update(obs["max_coords_obs"])
            self.target_normalizer.update(obs["mimic_target_poses"])

    def reset(self, dones=None, hidden_state=None) -> None:
        del dones, hidden_state

    def fsq_diagnostics(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            _, codes = self._features(obs)
            return {
                "fsq/perplexity": self.quantizer.calculate_perplexity(codes),
                "fsq/code_abs_mean": codes.abs().mean(),
                "fsq/code_saturation": (
                    codes.abs() >= self.quantizer.half_width.max()
                ).float().mean(),
            }


class FSQSACModel(BaseModel):
    """ProtoMotions evaluator/checkpoint wrapper around SAC actor and twin critic."""

    config: FSQSACModelConfig

    def __init__(self, config: FSQSACModelConfig, obs: TensorDict, action_dim: int):
        super().__init__(config)
        self.actor = FSQSACActor(obs, action_dim, config)
        self.critic = SACCriticModel(
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

