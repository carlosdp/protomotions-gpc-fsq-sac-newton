from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from gpc_fsq_sac.cli import (
    build_train_command,
    compare_results,
    create_parser,
    parse_evaluation_output,
)
from gpc_fsq_sac.constants import MOTION_COUNT, MOTION_MANIFEST_SHA256
from gpc_fsq_sac.fixture import manifest_from_payload


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
