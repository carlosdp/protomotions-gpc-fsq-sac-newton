from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from gpc_fsq_sac.cli import build_train_command, compare_results
from gpc_fsq_sac.constants import MOTION_COUNT, MOTION_MANIFEST_SHA256
from gpc_fsq_sac.fixture import manifest_from_payload


def train_args(tmp_path: Path, algorithm: str) -> argparse.Namespace:
    return argparse.Namespace(
        algorithm=algorithm,
        experiment_name=None,
        seed=3,
        motion_file=tmp_path / "fixture.pt",
        num_envs=64,
        batch_size=8192,
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

