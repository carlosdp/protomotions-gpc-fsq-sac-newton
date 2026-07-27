"""Matched ProtoMotions PPO-FSQ baseline for the controlled comparison."""

from gpc_fsq_sac import task

terrain_config = task.terrain_config
scene_lib_config = task.scene_lib_config
motion_lib_config = task.motion_lib_config
configure_robot_and_simulator = task.configure_robot_and_simulator
apply_inference_overrides = task.apply_inference_overrides


def env_config(robot_cfg, args):
    # Public ProtoMotions PPO emits an unsquashed Gaussian action; the task applies tanh once.
    return task.build_env_config(
        robot_cfg,
        action_transform="tanh",
        train_motion_id=getattr(args, "train_motion_id", None),
        fixed_starts=getattr(args, "fixed_starts", False),
    )


def agent_config(robot_config, env_config, args):
    del env_config
    return task.build_ppo_agent_config(robot_config, args)
