from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator

from gpc_fsq_sac.cli import (
    build_eval_command,
    build_train_command,
    compare_results,
    create_parser,
    parse_evaluation_output,
)
from gpc_fsq_sac.agent import FSQSACAgent, one_sided_conservative_q_loss
from gpc_fsq_sac.config import FSQSACModelConfig
from gpc_fsq_sac.constants import MOTION_COUNT, MOTION_MANIFEST_SHA256
from gpc_fsq_sac.evaluator import NewtonMimicEvaluator
from gpc_fsq_sac.fixture import manifest_from_payload
from gpc_fsq_sac.model import FSQSACModel
from gpc_fsq_sac.task import compute_sac_tracking_termination, evaluator_config


def train_args(tmp_path: Path, algorithm: str) -> argparse.Namespace:
    return argparse.Namespace(
        algorithm=algorithm,
        experiment_name=None,
        checkpoint=None,
        seed=3,
        motion_file=tmp_path / "fixture.pt",
        num_envs=64,
        rollout_steps=24,
        batch_size=768,
        training_steps=98_304,
        use_wandb=True,
        target_entropy_scale=0.167,
        replay_buffer_size=262_144,
        replay_memory_limit_gib=9.0,
        replay_warmup_transitions=0,
        sac_actor_start_epoch=0,
        sac_reset_alpha_on_load=False,
        sac_num_mini_batches=13,
        sac_policy_frequency=1,
        sac_actor_learning_rate=2e-4,
        sac_critic_learning_rate=2e-4,
        sac_conservative_q_coef=0.0,
        sac_conservative_q_batch_size=8192,
        sac_fixed_std=None,
        sac_fixed_physical_std=None,
        sac_reset_std_on_load=False,
        sac_reference_residual_actions=False,
        sac_reference_residual_action_scale=0.1,
        sac_reference_action_gain=1.0,
        sac_reference_action_time_offset_steps=1,
        sac_no_fsq=False,
        sac_freeze_normalization=False,
        sac_disable_tracking_termination=False,
        sac_tracking_termination_threshold=None,
        sac_min_log_std=-20.0,
        sac_max_log_std=2.0,
        sac_actor_trust_region_coef=0.0,
        sac_actor_reference_tau=0.01,
        sac_action_bounds_from_motion=False,
        sac_action_bounds_train_motion_only=False,
        sac_action_bounds_margin=0.05,
        sac_action_bounds_symmetric=True,
        sac_diagnostic_batch_size=1024,
        sac_diagnostic_every=10,
        eval_every=200,
        train_motion_id=None,
        fixed_starts=False,
        overrides=[],
    )


def test_sac_command_is_installed_dependency_workflow(tmp_path):
    command = build_train_command(train_args(tmp_path, "sac"))
    joined = " ".join(command)
    assert "-m protomotions.train_agent" in joined
    assert "--robot-name soma23" in joined
    assert "--simulator newton" in joined
    assert "soma23_fsq_sac.py" in joined
    assert "agent.target_entropy_scale=0.167" in joined
    assert "ProtoMotions" not in joined


def test_full_evaluation_clears_checkpoint_motion_filter(tmp_path):
    checkpoint = tmp_path / "score_based.ckpt"
    motion_file = tmp_path / "fixture.pt"
    args = SimpleNamespace(
        checkpoint=checkpoint,
        motion_file=motion_file,
    )

    command = build_eval_command(args)

    override_index = command.index("--overrides")
    assert command[override_index + 1 :] == [
        "agent.evaluator.evaluation_motion_ids=[]",
        "agent.evaluator.progressive_seed_motion_ids=[]",
    ]


def test_sac_curriculum_checkpoint_is_explicit(tmp_path):
    args = train_args(tmp_path, "sac")
    args.checkpoint = tmp_path / "stage-one.ckpt"
    command = build_train_command(args)
    checkpoint_index = command.index("--checkpoint")
    assert command[checkpoint_index + 1] == str(args.checkpoint.resolve())


def test_critic_uses_canonical_action_coordinates():
    obs = TensorDict(
        {
            "max_coords_obs": torch.zeros(2, 8),
            "mimic_target_poses": torch.zeros(2, 12),
        },
        batch_size=[2],
    )
    config = FSQSACModelConfig(
        num_fsq_scalars=4,
        encoder_hidden_dims=(8,),
        decoder_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        normalize_observations=False,
    )
    model = FSQSACModel(config=config, obs=obs, action_dim=3)
    bias = torch.tensor([0.2, -0.3, 0.1])
    action_range = torch.tensor([0.1, 0.25, 0.5])
    model.actor.action_bias.copy_(bias)
    model.actor.action_range.copy_(action_range)
    model.critic.set_action_scaling(bias, action_range)
    canonical = torch.tensor([[0.5, -0.5, 1.0], [-1.0, 0.0, 0.25]])
    physical = bias + action_range * canonical
    assert torch.allclose(
        model.critic.normalize_actions(obs, physical),
        canonical,
    )
    diagnostics = model.actor.policy_diagnostics(obs)
    assert "policy/target_action_sensitivity" in diagnostics


def test_sac_actor_reference_mask_selects_only_source_motions():
    obs = TensorDict(
        {
            "max_coords_obs": torch.zeros(4, 8),
            "mimic_target_poses": torch.zeros(4, 12),
            "sac_motion_id": torch.tensor([[3], [43], [3], [9]]),
        },
        batch_size=[4],
    )
    config = FSQSACModelConfig(
        num_fsq_scalars=4,
        encoder_hidden_dims=(8,),
        decoder_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        normalize_observations=False,
        actor_reference_motion_ids=[3],
    )
    model = FSQSACModel(config=config, obs=obs, action_dim=3)

    assert torch.equal(
        model.critic.actor_reference_mask(obs),
        torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
    )


def test_sac_motion_specific_critic_selects_q_without_actor_id_input():
    obs = TensorDict(
        {
            "max_coords_obs": torch.zeros(3, 8),
            "mimic_target_poses": torch.zeros(3, 12),
            "sac_motion_id": torch.tensor([[3], [1], [0]]),
        },
        batch_size=[3],
    )
    config = FSQSACModelConfig(
        num_fsq_scalars=4,
        encoder_hidden_dims=(8,),
        decoder_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        critic_num_motion_heads=4,
        normalize_observations=False,
    )
    model = FSQSACModel(config=config, obs=obs, action_dim=3)
    q_values = torch.tensor(
        [
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
        ]
    )

    assert torch.equal(
        model.critic.select_motion_q(obs, q_values),
        torch.tensor([[13.0], [21.0], [30.0]]),
    )
    assert "sac_motion_id" not in config.in_keys


def test_reference_residual_actor_starts_at_motion_target():
    reference_action = torch.tensor(
        [[0.2, -0.1, 0.4], [-0.3, 0.25, 0.1]]
    )
    obs = TensorDict(
        {
            "max_coords_obs": torch.zeros(2, 8),
            "mimic_target_poses": torch.zeros(2, 12),
            "sac_reference_action": reference_action,
        },
        batch_size=[2],
    )
    config = FSQSACModelConfig(
        num_fsq_scalars=4,
        encoder_hidden_dims=(8,),
        decoder_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        normalize_observations=False,
        reference_residual_actions=True,
        reference_residual_action_scale=0.1,
    )
    model = FSQSACModel(config=config, obs=obs, action_dim=3)
    action = model.actor(obs)

    assert torch.allclose(action, reference_action, atol=1e-5)
    residual = torch.tensor([[0.05, -0.1, 0.0], [-0.05, 0.0, 0.1]])
    assert torch.allclose(
        model.critic.normalize_actions(obs, reference_action + residual),
        residual / 0.1,
    )


def test_reference_action_gain_blends_nominal_and_motion_target():
    reference_action = torch.tensor([[0.2, -0.4, 0.6]])
    obs = TensorDict(
        {
            "max_coords_obs": torch.zeros(1, 8),
            "mimic_target_poses": torch.zeros(1, 12),
            "sac_reference_action": reference_action,
        },
        batch_size=[1],
    )
    config = FSQSACModelConfig(
        num_fsq_scalars=4,
        encoder_hidden_dims=(8,),
        decoder_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        normalize_observations=False,
        reference_residual_actions=True,
        reference_action_gain=0.5,
    )
    model = FSQSACModel(config=config, obs=obs, action_dim=3)

    assert torch.allclose(model.actor(obs), 0.5 * reference_action)


def test_sac_replay_observation_records_motion_identity():
    agent = object.__new__(FSQSACAgent)
    agent.config = SimpleNamespace(
        model=SimpleNamespace(reference_residual_actions=False)
    )
    agent.env = SimpleNamespace(
        motion_manager=SimpleNamespace(motion_ids=torch.tensor([3, 43]))
    )
    original = {"max_coords_obs": torch.zeros(2, 8)}

    enriched = agent.add_agent_info_to_obs(original)

    assert "sac_motion_id" not in original
    assert torch.equal(enriched["sac_motion_id"], torch.tensor([[3], [43]]))


def test_ppo_command_uses_same_requested_interaction_budget(tmp_path):
    sac = build_train_command(train_args(tmp_path, "sac"))
    ppo = build_train_command(train_args(tmp_path, "ppo"))
    sac_steps = sac[sac.index("--training-max-steps") + 1]
    ppo_steps = ppo[ppo.index("--training-max-steps") + 1]
    assert sac_steps == ppo_steps == "98304"


def test_default_ppo_batch_divides_default_rollout():
    args = create_parser().parse_args(["train", "ppo"])
    command = build_train_command(args)
    assert command[command.index("--batch-size") + 1] == "6144"


def test_focused_sac_probe_sets_fixed_std_warmup_and_motion(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_fixed_std = 0.055
    args.replay_warmup_transitions = 262_144
    args.sac_num_mini_batches = 2
    args.sac_policy_frequency = 2
    args.train_motion_id = 43
    args.fixed_starts = True
    command = build_train_command(args)
    overrides = command[command.index("--overrides") + 1 :]

    assert "agent.model.initial_std=0.055" in overrides
    assert "agent.model.learn_std=false" in overrides
    assert "agent.replay_warmup_transitions=262144" in overrides
    assert "agent.num_mini_batches=2" in overrides
    assert "agent.policy_frequency=2" in overrides
    assert "agent.evaluator.evaluation_motion_ids=[43]" in overrides
    assert "env.motion_manager.init_start_prob=1.0" in overrides
    exclusion = next(
        value
        for value in overrides
        if value.startswith("env.motion_manager.exclude_motion_ids=")
    )
    assert "43" not in exclusion.removeprefix(
        "env.motion_manager.exclude_motion_ids="
    ).strip("[]").split(", ")


def test_sac_stability_controls_are_explicit_overrides(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_actor_start_epoch = 100
    args.sac_actor_learning_rate = 5e-5
    args.sac_actor_trust_region_coef = 10.0
    args.sac_actor_reference_tau = 0.02
    args.sac_conservative_q_coef = 0.5
    args.sac_conservative_q_batch_size = 4096
    args.sac_freeze_normalization = True
    command = build_train_command(args)
    overrides = command[command.index("--overrides") + 1 :]

    assert "agent.actor_start_training_epoch=100" in overrides
    assert "agent.actor_learning_rate=5e-05" in overrides
    assert "agent.model.actor_trust_region_coef=10.0" in overrides
    assert "agent.model.actor_reference_tau=0.02" in overrides
    assert "agent.conservative_q_coef=0.5" in overrides
    assert "agent.conservative_q_batch_size=4096" in overrides
    assert "agent.model.freeze_normalization=true" in overrides


def test_sac_conservative_q_guard_penalizes_largest_ood_family():
    data_q = torch.tensor([[1.0], [1.0]])
    policy_q = torch.tensor([[1.4], [1.2]])
    random_q = torch.tensor([[-4.0], [-5.0]])

    loss = one_sided_conservative_q_loss(data_q, policy_q, random_q)

    assert loss.item() == pytest.approx(0.3)
    assert one_sided_conservative_q_loss(
        data_q,
        torch.zeros_like(data_q),
        torch.zeros_like(data_q),
    ).item() == 0.0


def test_no_fsq_probe_is_explicit(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_no_fsq = True
    command = build_train_command(args)
    overrides = command[command.index("--overrides") + 1 :]
    assert "agent.model.use_fsq=false" in overrides


def test_sac_curriculum_can_reset_exploration_std(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_fixed_std = 0.1
    args.sac_reset_std_on_load = True
    command = build_train_command(args)
    overrides = command[command.index("--overrides") + 1 :]
    assert "agent.model.initial_std=0.1" in overrides
    assert "agent.model.learn_std=false" in overrides
    assert "agent.model.reset_std_on_load=true" in overrides


def test_sac_vector_std_is_calibrated_in_physical_action_units(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_action_bounds_from_motion = True
    args.sac_fixed_physical_std = 0.03
    command = build_train_command(args)
    overrides = command[command.index("--overrides") + 1 :]
    assert "agent.model.physical_action_std=0.03" in overrides
    assert "agent.model.learn_std=false" in overrides

    obs = TensorDict(
        {
            "max_coords_obs": torch.zeros(2, 8),
            "mimic_target_poses": torch.zeros(2, 12),
        },
        batch_size=[2],
    )
    actor = FSQSACModel(
        config=FSQSACModelConfig(
            num_fsq_scalars=4,
            encoder_hidden_dims=(8,),
            decoder_hidden_dims=(8,),
            critic_hidden_dims=(8,),
            normalize_observations=False,
            learn_std=False,
        ),
        obs=obs,
        action_dim=3,
    ).actor
    actor.action_range.copy_(torch.tensor([0.02, 0.2, 0.6]))
    actor.set_physical_action_std(0.03)
    assert torch.allclose(
        actor.output_std * actor.action_range,
        torch.full((3,), 0.03),
    )


def test_sac_vector_std_requires_explicit_action_range(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_fixed_physical_std = 0.03
    with pytest.raises(ValueError, match="requires --sac-action-bounds"):
        build_train_command(args)


def test_sac_vector_std_accepts_reference_residual_range(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_fixed_physical_std = 0.03
    args.sac_reference_residual_actions = True
    command = build_train_command(args)
    overrides = command[command.index("--overrides") + 1 :]

    assert "agent.model.reference_residual_actions=true" in overrides
    assert "agent.model.reference_residual_action_scale=0.1" in overrides
    assert "agent.model.reference_action_gain=1.0" in overrides
    assert "agent.model.reference_action_time_offset_steps=1" in overrides
    assert "agent.model.physical_action_std=0.03" in overrides


def test_sac_scalar_and_physical_std_are_mutually_exclusive(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_action_bounds_from_motion = True
    args.sac_fixed_std = 0.1
    args.sac_fixed_physical_std = 0.03
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_train_command(args)


def test_sac_curriculum_can_reset_temperature(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_reset_alpha_on_load = True
    command = build_train_command(args)
    overrides = command[command.index("--overrides") + 1 :]
    assert "agent.reset_alpha_on_load=true" in overrides


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("true", True), ("false", False)],
)
def test_config_boolean_coercion(value, expected):
    assert FSQSACAgent._as_bool(value) is expected


def test_target_entropy_uses_environment_action_coordinates():
    assert FSQSACAgent.target_entropy_for_action_scale(-4.0, -7.5) == -11.5
    assert FSQSACAgent.target_entropy_for_action_scale(-4.0, 0.0) == -4.0


def test_sac_curriculum_reapplies_current_action_bounds_after_load(monkeypatch):
    from protomotions.agents.base_agent.agent import BaseAgent

    monkeypatch.setattr(
        BaseAgent,
        "_after_load_model_state_dict",
        lambda _self, _state_dict: None,
    )
    agent = FSQSACAgent.__new__(FSQSACAgent)
    agent.config = SimpleNamespace(
        model=SimpleNamespace(
            action_bounds_from_motion=True,
            reset_std_on_load=False,
        )
    )
    agent.model = object()
    applied = []
    agent._set_motion_action_bounds = lambda model: applied.append(model)

    agent._after_load_model_state_dict({})

    assert applied == [agent.model]


def test_sac_curriculum_reapplies_current_optimizer_learning_rates():
    class Optimizer:
        def __init__(self, learning_rate: float):
            self.param_groups = [{"lr": learning_rate}, {"lr": learning_rate}]

    agent = object.__new__(FSQSACAgent)
    agent.config = SimpleNamespace(
        actor_learning_rate=3e-5,
        critic_learning_rate=1e-4,
        alpha_learning_rate=2e-5,
    )
    agent.algorithm = SimpleNamespace(
        actor_optimizer=Optimizer(1e-5),
        critic_optimizer=Optimizer(2e-4),
        alpha_optimizer=Optimizer(9e-4),
    )

    agent._apply_configured_optimizer_learning_rates()

    assert {
        group["lr"] for group in agent.algorithm.actor_optimizer.param_groups
    } == {3e-5}
    assert {
        group["lr"] for group in agent.algorithm.critic_optimizer.param_groups
    } == {1e-4}
    assert {
        group["lr"] for group in agent.algorithm.alpha_optimizer.param_groups
    } == {2e-5}


def test_motion_calibrated_action_bounds_are_explicit(tmp_path):
    args = train_args(tmp_path, "sac")
    args.train_motion_id = 43
    args.sac_action_bounds_from_motion = True
    args.sac_action_bounds_train_motion_only = True
    args.sac_action_bounds_margin = 0.02
    command = build_train_command(args)
    overrides = command[command.index("--overrides") + 1 :]

    assert "agent.model.action_bounds_from_motion=true" in overrides
    assert "agent.model.action_bounds_motion_id=43" in overrides
    assert "agent.model.action_bounds_margin=0.02" in overrides
    assert "agent.model.action_bounds_symmetric=true" in overrides


def test_motion_subset_action_bounds_are_configurable():
    config = FSQSACModelConfig(action_bounds_motion_ids=[3, 9, 20, 30, 43, 45])
    assert config.action_bounds_motion_ids == [3, 9, 20, 30, 43, 45]


def test_motion_only_action_bounds_require_a_training_motion(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_action_bounds_from_motion = True
    args.sac_action_bounds_train_motion_only = True
    with pytest.raises(ValueError, match="requires --train-motion-id"):
        build_train_command(args)


def test_motion_only_action_bounds_require_calibration(tmp_path):
    args = train_args(tmp_path, "sac")
    args.train_motion_id = 43
    args.sac_action_bounds_train_motion_only = True
    with pytest.raises(ValueError, match="requires --sac-action-bounds-from-motion"):
        build_train_command(args)


def test_sac_can_relax_training_termination_for_dense_recovery(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_tracking_termination_threshold = 5.0
    command = build_train_command(args)
    threshold_index = command.index("--sac-tracking-termination-threshold")

    assert command[threshold_index + 1] == "5.0"


def test_sac_tracking_termination_uses_configured_threshold():
    reference = torch.zeros(2, 2, 3)
    current = reference.clone()
    current[0, 0, 0] = 1.5
    current[1, 0, 0] = 2.5

    terminated = compute_sac_tracking_termination(
        current,
        reference,
        max_error_threshold=2.0,
    )

    assert terminated.tolist() == [False, True]


def test_sac_can_disable_training_termination_for_dense_recovery(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_disable_tracking_termination = True
    command = build_train_command(args)
    assert "--sac-disable-tracking-termination" in command


def test_sac_rejects_conflicting_training_termination_controls(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_disable_tracking_termination = True
    args.sac_tracking_termination_threshold = 5.0

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_train_command(args)


def test_focused_evaluator_preserves_full_eval_environment_assignment():
    evaluator = NewtonMimicEvaluator.__new__(NewtonMimicEvaluator)
    evaluator.config = SimpleNamespace(evaluation_motion_ids=[43])
    evaluator.agent = SimpleNamespace(
        num_envs=512,
        motion_lib=SimpleNamespace(num_motions=lambda: MOTION_COUNT),
    )
    evaluator.fabric = SimpleNamespace(device=torch.device("cpu"))

    batches = evaluator._build_eval_batches()

    assert len(batches) == 1
    env_ids, motion_ids = batches[0]
    assert env_ids.tolist() == [43]
    assert motion_ids.tolist() == [43]


def test_focused_evaluator_stabilizes_only_inactive_sac_actions():
    evaluator = NewtonMimicEvaluator.__new__(NewtonMimicEvaluator)
    evaluator.agent = SimpleNamespace(num_envs=6)
    evaluator.fabric = SimpleNamespace(device=torch.device("cpu"))
    actions = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    reference_actions = -actions
    obs = TensorDict(
        {"sac_reference_action": reference_actions},
        batch_size=[6],
    )

    stabilized = evaluator._stabilize_inactive_actions(
        actions,
        obs,
        torch.tensor([1, 4]),
    )

    assert torch.equal(stabilized[[1, 4]], actions[[1, 4]])
    assert torch.equal(
        stabilized[[0, 2, 3, 5]],
        reference_actions[[0, 2, 3, 5]],
    )
    assert torch.equal(actions, torch.arange(12).reshape(6, 2))


def test_evaluator_reports_per_motion_metrics(monkeypatch):
    evaluator = NewtonMimicEvaluator.__new__(NewtonMimicEvaluator)
    evaluator._motion_failed = torch.tensor([False, True, False])
    evaluator._eval_mask = torch.tensor([True, True, False])
    evaluator._component_value_sum = {
        "gt_error": torch.tensor([1.0, 6.0, 0.0]),
    }
    evaluator._component_value_max = {
        "gt_error": torch.tensor([0.4, 4.0, float("-inf")]),
    }
    evaluator._component_step_count = {
        "gt_error": torch.tensor([4, 3, 0]),
    }
    monkeypatch.setattr(
        MimicEvaluator,
        "process_eval_results",
        lambda self: ({"eval/success_rate": 0.5}, 0.5, 2),
    )

    metrics, score, count = evaluator.process_eval_results()

    assert score == 0.5
    assert count == 2
    assert metrics["eval/motion_0/success"] == 1.0
    assert metrics["eval/motion_0/gt_error_mean"] == pytest.approx(0.25)
    assert metrics["eval/motion_0/gt_error_max"] == pytest.approx(0.4)
    assert metrics["eval/motion_1/success"] == 0.0
    assert metrics["eval/motion_1/gt_error_mean"] == pytest.approx(2.0)
    assert "eval/motion_2/success" not in metrics


def test_model_selection_breaks_success_ties_with_tracking_error():
    weaker = FSQSACAgent.model_selection_score(
        {"eval/gt_error/mean": torch.tensor(2.0)},
        0.0,
    )
    stronger = FSQSACAgent.model_selection_score(
        {"eval/gt_error/mean": torch.tensor(1.0)},
        0.0,
    )
    one_success = FSQSACAgent.model_selection_score(
        {"eval/gt_error/mean": torch.tensor(100.0)},
        0.5,
    )

    assert stronger > weaker
    assert one_success > stronger


def test_full_corpus_stage_outranks_seed_stage_for_model_selection():
    perfect_seed = FSQSACAgent.model_selection_score(
        {
            "eval/gt_error/mean": 0.01,
            "eval/progressive_stage": 0.0,
        },
        1.0,
    )
    weak_full_corpus = FSQSACAgent.model_selection_score(
        {
            "eval/gt_error/mean": 2.0,
            "eval/progressive_stage": 1.0,
        },
        0.1,
    )

    assert weak_full_corpus > perfect_seed


def test_progressive_curriculum_initializes_seed_sampling_without_exclusions():
    updated_weights = []
    motion_manager = SimpleNamespace(
        motion_weights=torch.ones(4),
        excluded_motion_ids=None,
        update_sampling_weights=lambda weights: updated_weights.append(
            weights.clone()
        ),
    )
    evaluator = NewtonMimicEvaluator.__new__(NewtonMimicEvaluator)
    evaluator.config = SimpleNamespace(
        progressive_seed_motion_ids=[1, 3],
        progressive_expand_success_rate=1.0,
        progressive_expand_consecutive_evals=1,
        progressive_actor_pause_epochs=25,
        evaluation_motion_ids=[],
    )
    evaluator.agent = SimpleNamespace(
        motion_lib=SimpleNamespace(num_motions=lambda: 4),
        env=SimpleNamespace(motion_manager=motion_manager),
    )

    evaluator._initialize_progressive_curriculum()

    assert evaluator.config.evaluation_motion_ids == [1, 3]
    assert torch.equal(
        updated_weights[-1],
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )


def test_progressive_curriculum_unlocks_all_motions_in_same_process():
    updated_weights = []
    motion_manager = SimpleNamespace(
        motion_weights=torch.tensor([0.0, 1.0, 0.0, 1.0]),
        update_sampling_weights=lambda weights: updated_weights.append(
            weights.clone()
        ),
    )
    evaluator = NewtonMimicEvaluator.__new__(NewtonMimicEvaluator)
    evaluator.config = SimpleNamespace(
        progressive_seed_motion_ids=[1, 3],
        progressive_expand_success_rate=1.0,
        progressive_expand_consecutive_evals=1,
        progressive_actor_pause_epochs=25,
        evaluation_motion_ids=[1, 3],
    )
    evaluator.agent = SimpleNamespace(
        current_epoch=750,
        env=SimpleNamespace(motion_manager=motion_manager),
    )
    evaluator._motion_failed = torch.tensor([False, False, False, False])
    evaluator._eval_mask = torch.tensor([False, True, False, True])
    evaluator._progressive_expanded = False
    evaluator._progressive_success_streak = 0
    evaluator._progressive_expansion_epoch = None
    evaluator._progressive_actor_resume_epoch = None

    expanded = evaluator._maybe_expand_progressive_curriculum()

    assert expanded is True
    assert evaluator._progressive_expanded is True
    assert evaluator._progressive_expansion_epoch == 750
    assert evaluator._progressive_actor_resume_epoch == 775
    assert evaluator.agent._curriculum_actor_resume_epoch == 775
    assert evaluator.config.evaluation_motion_ids == []
    assert torch.equal(updated_weights[-1], torch.ones(4))


def test_evaluator_retains_native_failure_weighted_curriculum():
    config = evaluator_config(eval_metrics_every=250)

    assert (
        config.motion_weights_rules.motion_weights_update_success_discount
        == 0.999
    )
    assert (
        config.motion_weights_rules.motion_weights_update_failure_discount
        == 0.999
    )


def test_continuous_error_curriculum_prioritizes_harder_motion():
    evaluator = NewtonMimicEvaluator.__new__(NewtonMimicEvaluator)
    evaluator.config = SimpleNamespace(
        continuous_error_curriculum=True,
        continuous_error_curriculum_start_epoch=0,
        continuous_error_curriculum_min_success_rate=0.0,
        continuous_error_curriculum_alpha=0.25,
        continuous_error_curriculum_min_relative_weight=0.5,
        continuous_error_curriculum_max_relative_weight=2.0,
    )
    evaluator._motion_failed = torch.tensor([True, True, False])
    evaluator._eval_mask = torch.tensor([True, True, False])
    evaluator._component_value_sum = {
        "gt_error": torch.tensor([1.0, 6.0, 0.0]),
    }
    evaluator._component_step_count = {
        "gt_error": torch.tensor([4, 3, 0]),
    }
    updated_weights = []
    evaluator.agent = SimpleNamespace(
        current_epoch=25,
        env=SimpleNamespace(
            motion_manager=SimpleNamespace(
                motion_weights=torch.tensor([1.0, 1.0, 0.0]),
                update_sampling_weights=lambda weights: updated_weights.append(
                    weights.clone()
                ),
            )
        )
    )
    evaluator._save_failed_motions = lambda failed, epoch: None

    evaluator._update_motion_sampling_weights()

    assert len(updated_weights) == 1
    assert updated_weights[0][0] < 1.0
    assert updated_weights[0][1] > 1.0
    assert updated_weights[0][2] == 0.0
    assert updated_weights[0][:2].mean() == pytest.approx(1.0)
    assert evaluator._continuous_sampling_weights[1] > (
        evaluator._continuous_sampling_weights[0]
    )


def test_continuous_error_curriculum_validates_weight_range():
    evaluator = NewtonMimicEvaluator.__new__(NewtonMimicEvaluator)
    evaluator.config = SimpleNamespace(
        continuous_error_curriculum=True,
        continuous_error_curriculum_start_epoch=0,
        continuous_error_curriculum_min_success_rate=0.0,
        continuous_error_curriculum_alpha=0.25,
        continuous_error_curriculum_min_relative_weight=2.0,
        continuous_error_curriculum_max_relative_weight=1.0,
    )
    evaluator._motion_failed = torch.tensor([True])
    evaluator._eval_mask = torch.tensor([True])
    evaluator._component_value_sum = {"gt_error": torch.tensor([1.0])}
    evaluator._component_step_count = {"gt_error": torch.tensor([1])}
    evaluator.agent = SimpleNamespace(current_epoch=0)

    with pytest.raises(ValueError, match="0 < min <= max"):
        evaluator._update_motion_sampling_weights()


def test_continuous_error_curriculum_honors_burn_in(monkeypatch):
    evaluator = NewtonMimicEvaluator.__new__(NewtonMimicEvaluator)
    evaluator.config = SimpleNamespace(
        continuous_error_curriculum=True,
        continuous_error_curriculum_start_epoch=500,
    )
    evaluator.agent = SimpleNamespace(current_epoch=499)
    native_updates = []
    monkeypatch.setattr(
        MimicEvaluator,
        "_update_motion_sampling_weights",
        lambda self: native_updates.append(self.agent.current_epoch),
    )

    evaluator._update_motion_sampling_weights()

    assert native_updates == [499]


def test_continuous_error_curriculum_honors_minimum_success_rate(
    monkeypatch,
):
    evaluator = NewtonMimicEvaluator.__new__(NewtonMimicEvaluator)
    evaluator.config = SimpleNamespace(
        continuous_error_curriculum=True,
        continuous_error_curriculum_start_epoch=500,
        continuous_error_curriculum_min_success_rate=0.5,
    )
    evaluator.agent = SimpleNamespace(current_epoch=500)
    evaluator._motion_failed = torch.tensor([True, True, False])
    evaluator._eval_mask = torch.tensor([True, True, False])
    native_updates = []
    monkeypatch.setattr(
        MimicEvaluator,
        "_update_motion_sampling_weights",
        lambda self: native_updates.append(self.agent.current_epoch),
    )

    evaluator._update_motion_sampling_weights()

    assert native_updates == [500]
    assert evaluator._continuous_sampling_weights == {}


def test_ppo_rejects_non_divisible_batch(tmp_path):
    args = train_args(tmp_path, "ppo")
    args.batch_size = 8192
    with pytest.raises(ValueError, match="not divisible"):
        build_train_command(args)


def test_manifest_payload_contract():
    names = [f"motion_{index:02d}" for index in range(MOTION_COUNT)]
    assert manifest_from_payload({"motion_files": tuple(names)}) == names
    assert len(MOTION_MANIFEST_SHA256) == 64


def test_compare_rejects_unequal_budgets(tmp_path):
    sac = tmp_path / "sac.json"
    ppo = tmp_path / "ppo.json"
    sac.write_text(json.dumps({"step_count": 10, "num_evaluated": 61}))
    ppo.write_text(json.dumps({"step_count": 11, "num_evaluated": 61}))
    with pytest.raises(ValueError, match="unequal"):
        compare_results(sac, ppo)


def test_compare_requires_all_motions(tmp_path):
    sac = tmp_path / "sac.json"
    ppo = tmp_path / "ppo.json"
    sac.write_text(json.dumps({"step_count": 10, "num_evaluated": 60}))
    ppo.write_text(json.dumps({"step_count": 10, "num_evaluated": 60}))
    with pytest.raises(ValueError, match="61"):
        compare_results(sac, ppo)


def test_evaluation_parser_requires_all_fixed_order_motions(tmp_path, monkeypatch):
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")

    class TorchStub:
        @staticmethod
        def load(*_args, **_kwargs):
            return {"epoch": 100, "step_count": 1_228_800}

    monkeypatch.setitem(__import__("sys").modules, "torch", TorchStub)
    output = "\n".join(
        [
            "  eval/success_rate: 0.250000",
            "  eval/num_evaluated: 61.000000",
            "  Items Evaluated: 61",
            "  Overall Score: 0.250000",
        ]
    )
    parsed = parse_evaluation_output(output, checkpoint)
    assert parsed["num_evaluated"] == 61
    assert parsed["fixed_order_motion_ids"] == list(range(61))
    assert parsed["epoch"] == 100
    assert parsed["step_count"] == 1_228_800
