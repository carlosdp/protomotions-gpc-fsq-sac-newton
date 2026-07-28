"""SOMA23 FSQ tracker using the released rsl_rl SAC learner."""

from gpc_fsq_sac import task

terrain_config = task.terrain_config
scene_lib_config = task.scene_lib_config
motion_lib_config = task.motion_lib_config
configure_robot_and_simulator = task.configure_robot_and_simulator
apply_inference_overrides = task.apply_inference_overrides


def additional_experiment_arguments(parser):
    parser.add_argument(
        "--sac-disable-tracking-termination",
        action="store_true",
    )
    parser.add_argument(
        "--sac-tracking-termination-threshold",
        type=float,
        default=0.5,
    )


def env_config(robot_cfg, args):
    # SAC already applies its tanh squash. The environment must not apply a second tanh.
    return task.build_env_config(
        robot_cfg,
        action_transform=None,
        train_motion_id=getattr(args, "train_motion_id", None),
        fixed_starts=getattr(args, "fixed_starts", False),
        disable_tracking_termination=getattr(
            args,
            "sac_disable_tracking_termination",
            False,
        ),
        tracking_termination_threshold=getattr(
            args,
            "sac_tracking_termination_threshold",
            0.5,
        ),
    )


def agent_config(robot_config, env_config, args):
    del env_config
    return task.build_sac_agent_config(robot_config, args)
