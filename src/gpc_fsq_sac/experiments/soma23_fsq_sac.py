"""SOMA23 FSQ tracker using the released rsl_rl SAC learner."""

from gpc_fsq_sac import task

terrain_config = task.terrain_config
scene_lib_config = task.scene_lib_config
motion_lib_config = task.motion_lib_config
configure_robot_and_simulator = task.configure_robot_and_simulator
apply_inference_overrides = task.apply_inference_overrides


def env_config(robot_cfg, args):
    # SAC already applies its tanh squash. The environment must not apply a second tanh.
    return task.build_env_config(robot_cfg, action_transform=None)


def agent_config(robot_config, env_config, args):
    del env_config
    return task.build_sac_agent_config(robot_config, args)
