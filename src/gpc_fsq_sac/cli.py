"""User-facing commands for fixture acquisition and controlled experiments."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from importlib import resources
from pathlib import Path

from .constants import (
    DEFAULT_NUM_ENVS,
    DEFAULT_ROLLOUT_STEPS,
    DEFAULT_TARGET_ENTROPY_SCALE,
    DEFAULT_TRAINING_STEPS,
)
from .fixture import default_motion_path, fetch_motion_file, verify_motion_manifest


def experiment_path(algorithm: str) -> Path:
    name = f"soma23_fsq_{algorithm}.py"
    return Path(str(resources.files("gpc_fsq_sac.experiments").joinpath(name)))


def build_train_command(args: argparse.Namespace) -> list[str]:
    experiment_name = args.experiment_name or (
        f"soma23_bones_seed_mini_fsq_{args.algorithm}_seed{args.seed}"
    )
    command = [
        sys.executable,
        "-m",
        "protomotions.train_agent",
        "--robot-name",
        "soma23",
        "--simulator",
        "newton",
        "--experiment-path",
        str(experiment_path(args.algorithm)),
        "--experiment-name",
        experiment_name,
        "--motion-file",
        str(args.motion_file.expanduser().resolve()),
        "--num-envs",
        str(args.num_envs),
        "--batch-size",
        str(args.batch_size),
        "--training-max-steps",
        str(args.training_steps),
        "--seed",
        str(args.seed),
        "--ngpu",
        "1",
        "--headless",
        "true",
    ]
    if args.use_wandb:
        command.append("--use-wandb")
    overrides = list(args.overrides)
    if args.algorithm == "sac":
        overrides.extend(
            [
                f"agent.target_entropy_scale={args.target_entropy_scale}",
                f"agent.replay_buffer_size={args.replay_buffer_size}",
                f"agent.replay_memory_limit_gib={args.replay_memory_limit_gib}",
            ]
        )
    if overrides:
        command.extend(["--overrides", *overrides])
    return command


def build_eval_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "protomotions.inference_agent",
        "--checkpoint",
        str(args.checkpoint.expanduser().resolve()),
        "--full-eval",
        "--headless",
        "--simulator",
        "newton",
        "--num-envs",
        "61",
        "--motion-file",
        str(args.motion_file.expanduser().resolve()),
    ]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpc-fsq")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch-motion")
    fetch.add_argument("--output", type=Path, default=default_motion_path())
    fetch.add_argument("--accept-bones-seed-license", action="store_true")

    verify = subparsers.add_parser("verify-motion")
    verify.add_argument("path", type=Path, nargs="?", default=default_motion_path())

    train = subparsers.add_parser("train")
    train.add_argument("algorithm", choices=["sac", "ppo"])
    train.add_argument("--motion-file", type=Path, default=default_motion_path())
    train.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    train.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    train.add_argument("--batch-size", type=int, default=8192)
    train.add_argument("--training-steps", type=int, default=DEFAULT_TRAINING_STEPS)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--experiment-name")
    train.add_argument("--use-wandb", action="store_true")
    train.add_argument("--target-entropy-scale", type=float, default=DEFAULT_TARGET_ENTROPY_SCALE)
    train.add_argument("--replay-buffer-size", type=int, default=262_144)
    train.add_argument("--replay-memory-limit-gib", type=float, default=9.0)
    train.add_argument("--overrides", nargs="*", default=[])
    train.add_argument("--print-command", action="store_true")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--motion-file", type=Path, default=default_motion_path())
    evaluate.add_argument("--print-command", action="store_true")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--sac", type=Path, required=True)
    compare.add_argument("--ppo", type=Path, required=True)
    compare.add_argument("--output", type=Path, default=Path("comparison.json"))
    return parser


def _read_evaluation(path: Path) -> dict:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "evaluation_final.json"
    return json.loads(path.read_text())


def compare_results(sac_path: Path, ppo_path: Path) -> dict:
    sac = _read_evaluation(sac_path)
    ppo = _read_evaluation(ppo_path)
    if sac["step_count"] != ppo["step_count"]:
        raise ValueError(
            f"unequal environment interaction budgets: "
            f"SAC={sac['step_count']} PPO={ppo['step_count']}"
        )
    if sac["num_evaluated"] != 61 or ppo["num_evaluated"] != 61:
        raise ValueError("both evaluations must contain all 61 fixed-order motions")
    shared_metrics = sorted(
        key
        for key in set(sac) & set(ppo)
        if key.startswith("eval/") and isinstance(sac[key], (int, float))
    )
    return {
        "scope": "small-corpus algorithm-integration comparison; not GPC corpus parity",
        "environment_interactions": sac["step_count"],
        "num_fixed_order_motions": 61,
        "sac": sac,
        "ppo": ppo,
        "delta_sac_minus_ppo": {
            key: sac[key] - ppo[key] for key in shared_metrics
        },
    }


def main() -> int:
    args = create_parser().parse_args()
    if args.command == "fetch-motion":
        metadata = fetch_motion_file(
            args.output,
            accept_license=args.accept_bones_seed_license,
        )
        print(json.dumps(metadata, indent=2))
        return 0
    if args.command == "verify-motion":
        print(json.dumps(verify_motion_manifest(args.path), indent=2))
        return 0
    if args.command == "compare":
        comparison = compare_results(args.sac, args.ppo)
        args.output.write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps(comparison, indent=2))
        return 0

    if args.command == "train":
        if args.rollout_steps != DEFAULT_ROLLOUT_STEPS:
            args.overrides.append(f"agent.num_steps={args.rollout_steps}")
        verify_motion_manifest(args.motion_file)
        command = build_train_command(args)
        os.environ.setdefault("WANDB_RUN_GROUP", "soma23-mini-fsq-sac-vs-ppo")
        os.environ.setdefault("WANDB_TAGS", "soma23-mini,fsq,newton,controlled-comparison")
    else:
        verify_motion_manifest(args.motion_file)
        command = build_eval_command(args)

    if args.print_command:
        print(shlex.join(command))
        return 0
    os.execvpe(sys.executable, command, os.environ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
