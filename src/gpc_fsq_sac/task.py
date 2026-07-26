"""Shared SOMA23 task and exact ProtoMotions PPO-FSQ baseline."""

from __future__ import annotations

import argparse

from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


def terrain_config(args: argparse.Namespace):
    del args
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    from protomotions.components.scene_lib import SceneLibConfig

    return SceneLibConfig(scene_file=getattr(args, "scenes_file", None))


def motion_lib_config(args: argparse.Namespace):
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def build_env_config(
    robot_cfg: RobotConfig,
    *,
    action_transform: str | None,
) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        contact_match_rew_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        mimic_tracking_rewards_factory,
        pow_rew_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        reset_grace_period=5,
        ref_respawn_offset=0.0,
        control_components={
            "mimic": MimicControlConfig(
                bootstrap_on_episode_end=True,
                future_steps=[1, 2, 5, 7, 12, 18, 25],
            )
        },
        observation_components={
            "max_coords_obs": max_coords_obs_factory(root_height_obs=True),
            "mimic_target_poses": mimic_target_poses_max_coords_factory(
                with_velocities=True,
            ),
        },
        termination_components={
            "tracking_error": tracking_error_term_factory(threshold=0.5),
        },
        reward_components={
            **mimic_tracking_rewards_factory(
                gt_weight=0.5,
                gr_weight=0.3,
                gv_weight=0.1,
                gav_weight=0.1,
                rh_weight=0.2,
                gt_coef=-100.0,
                gr_coef=-5.0,
                gv_coef=-0.5,
                gav_coef=-0.1,
                rh_coef=-100.0,
            ),
            "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
            "contact_match_rew": contact_match_rew_factory(
                weight=-0.1,
                zero_during_grace_period=True,
            ),
        },
        action_config=make_pd_action_config(
            robot_cfg,
            action_transform=action_transform,
        ),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    args: argparse.Namespace,
):
    del args
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )
    # SOMA23 can exceed Newton's generic 450-row default during broad contact.
    # 1024 gives >2x headroom over the observed 456-row requirement.
    simulator_cfg.sim.njmax = 1024
    robot_cfg.simulation_params.newton.njmax = 1024


def evaluator_config(eval_metrics_every: int):
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.envs.component_factories import (
        gr_error_factory,
        gt_error_factory,
        max_joint_error_factory,
    )

    return MimicEvaluatorConfig(
        _target_="gpc_fsq_sac.evaluator.NewtonMimicEvaluator",
        eval_metrics_every=eval_metrics_every,
        save_predicted_motion_lib_every=None,
        evaluation_components={
            "gt_error": gt_error_factory(threshold=0.5),
            "gr_error": gr_error_factory(),
            "max_joint_error": max_joint_error_factory(),
        },
        motion_weights_rules=MotionWeightsRulesConfig(
            motion_weights_update_success_discount=1.0,
            motion_weights_update_failure_discount=1.0,
        ),
    )


def build_sac_agent_config(
    robot_config: RobotConfig,
    args: argparse.Namespace,
):
    from .config import FSQSACAgentConfig, FSQSACModelConfig

    return FSQSACAgentConfig(
        model=FSQSACModelConfig(),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        replay_buffer_size=getattr(args, "sac_replay_buffer_size", 262_144),
        target_entropy_scale=getattr(args, "sac_target_entropy_scale", 0.167),
        evaluator=evaluator_config(eval_metrics_every=200),
    )


def build_ppo_agent_config(
    robot_config: RobotConfig,
    args: argparse.Namespace,
):
    """Reproduce ProtoMotions' public examples/experiments/mimic/fsq.py."""
    from protomotions.agents.base_agent.config import (
        MuonWithAuxAdamConfig,
        OptimizerConfig,
    )
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.common.fsq_config import FSQAutoEncoderConfig
    from protomotions.agents.ppo.config import (
        AdvantageNormalizationConfig,
        PPOActorConfig,
        PPOAgentConfig,
        PPOModelConfig,
    )

    encoder = MLPWithConcatConfig(
        in_keys=["mimic_target_poses"],
        out_keys=["latent"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=40,
        layers=[
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=256, activation="relu"),
        ],
    )
    decoder = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "latent"],
        out_keys=["mu"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=robot_config.number_of_actions,
        layers=[
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=256, activation="relu"),
        ],
    )
    fsq = FSQAutoEncoderConfig(
        num_fsq_levels=9,
        num_fsq_scalars=40,
        encoder_out_keys=["latent"],
        decoder_out_keys=["mu"],
        encoder=encoder,
        decoder=decoder,
    )
    actor = PPOActorConfig(
        mu_key="mu",
        in_keys=["max_coords_obs", "mimic_target_poses"],
        mu_model=fsq,
        num_out=robot_config.number_of_actions,
        actor_logstd=-2.9,
    )
    critic = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "mimic_target_poses"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )
    return PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=["max_coords_obs", "mimic_target_poses"],
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor,
            critic=critic,
            actor_optimizer=MuonWithAuxAdamConfig(
                lr=5e-4,
                weight_decay=0.01,
                momentum=0.95,
                adam_lr=1e-4,
                adam_betas=(0.9, 0.95),
                adam_eps=1e-10,
                adam_weight_decay=0.01,
            ),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        save_inference_checkpoint=True,
        evaluator=evaluator_config(eval_metrics_every=200),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True,
            shift_mean=True,
            use_ema=True,
        ),
    )


def apply_inference_overrides(
    robot_cfg,
    simulator_cfg,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args,
):
    del robot_cfg, simulator_cfg, agent_cfg, terrain_cfg, motion_lib_cfg, scene_lib_cfg, args
    env_cfg.termination_components = {}
    env_cfg.max_episode_length = 1_000_000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
