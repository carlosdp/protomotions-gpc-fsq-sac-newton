"""Configuration dataclasses for the released SAC integration."""

from __future__ import annotations

from dataclasses import dataclass, field

from protomotions.agents.base_agent.config import BaseAgentConfig, BaseModelConfig


@dataclass
class FSQSACModelConfig(BaseModelConfig):
    _target_: str = "gpc_fsq_sac.model.FSQSACModel"
    in_keys: list[str] = field(
        default_factory=lambda: ["max_coords_obs", "mimic_target_poses"]
    )
    out_keys: list[str] = field(default_factory=lambda: ["action", "mean_action"])
    num_fsq_levels: int = 9
    num_fsq_scalars: int = 40
    encoder_hidden_dims: tuple[int, ...] = (1024, 1024, 1024, 512, 256)
    decoder_hidden_dims: tuple[int, ...] = (1024, 1024, 1024, 512, 256)
    critic_hidden_dims: tuple[int, ...] = (1024, 1024, 1024, 1024)
    initial_std: float = 0.15
    normalize_observations: bool = True
    normalization_clip: float = 5.0


@dataclass
class FSQSACAgentConfig(BaseAgentConfig):
    _target_: str = "gpc_fsq_sac.agent.FSQSACAgent"
    model: FSQSACModelConfig = field(default_factory=FSQSACModelConfig)
    num_steps: int = 24
    num_mini_epochs: int = 1
    normalize_rewards: bool = False
    gamma: float = 0.97
    replay_buffer_size: int = 262_144
    replay_memory_limit_gib: float = 9.0
    n_steps: int = 5
    num_learning_epochs: int = 1
    num_mini_batches: int = 200
    mini_batch_size: int = 8192
    start_training_epoch: int = 1
    actor_learning_rate: float = 2e-4
    critic_learning_rate: float = 2e-4
    alpha_learning_rate: float = 2e-5
    tau: float = 0.003
    initial_alpha: float = 0.001
    auto_alpha: bool = True
    target_entropy_scale: float = 0.167
    policy_frequency: int = 1
    max_grad_norm: float = 1.0
    checkpoint_replay_buffer: bool = False
    save_last_checkpoint_every: int = 20
    save_epoch_checkpoint_every: int | None = 200
    save_inference_checkpoint: bool = True

