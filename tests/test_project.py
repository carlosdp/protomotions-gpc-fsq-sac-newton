from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from gpc_fsq_sac.cli import (
    build_train_command,
    compare_results,
    create_parser,
    parse_evaluation_output,
)
from gpc_fsq_sac.constants import MOTION_COUNT, MOTION_MANIFEST_SHA256
from gpc_fsq_sac.evaluator import NewtonMimicEvaluator
from gpc_fsq_sac.fixture import manifest_from_payload
from gpc_fsq_sac.task import apply_inference_overrides


def train_args(tmp_path: Path, algorithm: str) -> argparse.Namespace:
    return argparse.Namespace(
        algorithm=algorithm,
        experiment_name=None,
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
        sac_num_mini_batches=13,
        sac_policy_frequency=1,
        sac_actor_learning_rate=2e-4,
        sac_critic_learning_rate=2e-4,
        sac_fixed_std=None,
        sac_no_fsq=False,
        sac_ppo_actor_checkpoint=None,
        sac_freeze_normalization=False,
        sac_tracking_termination_threshold=None,
        sac_min_log_std=-20.0,
        sac_max_log_std=2.0,
        sac_actor_trust_region_coef=0.0,
        sac_actor_reference_tau=0.01,
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


def test_sac_phase_two_diagnostics_are_explicit_overrides(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_actor_start_epoch = 100
    args.sac_actor_learning_rate = 5e-5
    args.sac_actor_trust_region_coef = 10.0
    args.sac_actor_reference_tau = 0.02
    args.sac_ppo_actor_checkpoint = tmp_path / "ppo.ckpt"
    args.sac_freeze_normalization = True
    command = build_train_command(args)
    overrides = command[command.index("--overrides") + 1 :]

    assert "agent.actor_start_training_epoch=100" in overrides
    assert "agent.actor_learning_rate=5e-05" in overrides
    assert "agent.model.actor_trust_region_coef=10.0" in overrides
    assert "agent.model.actor_reference_tau=0.02" in overrides
    assert "agent.model.ppo_compatible_normalization=true" in overrides
    assert (
        f"agent.model.ppo_actor_checkpoint={(tmp_path / 'ppo.ckpt').resolve()}"
        in overrides
    )
    assert "agent.model.freeze_normalization=true" in overrides


def test_no_fsq_probe_is_explicit(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_no_fsq = True
    command = build_train_command(args)
    overrides = command[command.index("--overrides") + 1 :]
    assert "agent.model.use_fsq=false" in overrides


def test_sac_can_relax_training_termination_for_dense_recovery(tmp_path):
    args = train_args(tmp_path, "sac")
    args.sac_tracking_termination_threshold = 5.0
    command = build_train_command(args)
    overrides = command[command.index("--overrides") + 1 :]
    assert (
        "env.termination_components.tracking_error.static_params.threshold=5.0"
        in overrides
    )


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


def test_inference_checkpoint_does_not_reopen_ppo_warmstart():
    env_cfg = SimpleNamespace(
        termination_components={"tracking_error": object()},
        max_episode_length=1000,
        motion_manager=SimpleNamespace(
            resample_on_reset=False,
            init_start_prob=0.2,
        ),
    )
    agent_cfg = SimpleNamespace(
        model=SimpleNamespace(ppo_actor_checkpoint="/training-only/ppo.ckpt")
    )

    apply_inference_overrides(
        None,
        None,
        env_cfg,
        agent_cfg,
        None,
        None,
        None,
        None,
    )

    assert agent_cfg.model.ppo_actor_checkpoint is None


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
