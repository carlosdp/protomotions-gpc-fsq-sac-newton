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
    use_fsq: bool = True
    freeze_normalization: bool = False
    encoder_hidden_dims: tuple[int, ...] = (1024, 1024, 1024, 512, 256)
    decoder_hidden_dims: tuple[int, ...] = (1024, 1024, 1024, 512, 256)
    critic_hidden_dims: tuple[int, ...] = (1024, 1024, 1024, 1024)
    critic_num_motion_heads: int = 1
    initial_std: float = 0.15
    physical_action_std: float | None = None
    learn_std: bool = True
    reset_std_on_load: bool = False
    reference_residual_actions: bool = False
    reference_residual_action_scale: float = 0.1
    reference_action_gain: float = 1.0
    reference_action_time_offset_steps: int = 1
    min_log_std: float = -20.0
    max_log_std: float = 2.0
    actor_trust_region_coef: float = 0.0
    actor_reference_tau: float = 0.01
    actor_reference_motion_ids: list[int] | None = None
    action_bounds_from_motion: bool = False
    action_bounds_motion_id: int | None = None
    action_bounds_motion_ids: list[int] | None = None
    action_bounds_margin: float = 0.05
    action_bounds_symmetric: bool = True
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
    replay_warmup_transitions: int = 0
    actor_start_training_epoch: int = 0
    actor_learning_rate: float = 2e-4
    critic_learning_rate: float = 2e-4
    alpha_learning_rate: float = 2e-5
    tau: float = 0.003
    initial_alpha: float = 0.001
    auto_alpha: bool = True
    reset_alpha_on_load: bool = False
    target_entropy_scale: float = 0.167
    policy_frequency: int = 1
    max_grad_norm: float = 1.0
    conservative_q_coef: float = 0.0
    conservative_q_batch_size: int = 8192
    checkpoint_replay_buffer: bool = False
    diagnostic_batch_size: int = 1024
    diagnostic_every: int = 10
    save_last_checkpoint_every: int = 20
    save_epoch_checkpoint_every: int | None = 200
    save_inference_checkpoint: bool = True
