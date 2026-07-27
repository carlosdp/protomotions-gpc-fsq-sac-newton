"""User-facing commands for fixture acquisition and controlled experiments."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from importlib import resources
from pathlib import Path

from .constants import (
    MOTION_COUNT,
    DEFAULT_NUM_ENVS,
    DEFAULT_ROLLOUT_STEPS,
    DEFAULT_TARGET_ENTROPY_SCALE,
    DEFAULT_TRAINING_STEPS,
)
from .fixture import default_motion_path, fetch_motion_file, verify_motion_manifest


def experiment_path(algorithm: str) -> Path:
    name = f"soma23_fsq_{algorithm}.py"
    return Path(str(resources.files("gpc_fsq_sac.experiments").joinpath(name)))


def resolved_batch_size(args: argparse.Namespace) -> int:
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 8_192 if args.algorithm == "sac" else 6_144
    rollout_size = args.num_envs * getattr(args, "rollout_steps", DEFAULT_ROLLOUT_STEPS)
    if args.algorithm == "ppo" and rollout_size % batch_size:
        raise ValueError(
            "PPO requires num_envs * rollout_steps to be divisible by batch_size: "
            f"{args.num_envs} * {getattr(args, 'rollout_steps', DEFAULT_ROLLOUT_STEPS)} "
            f"= {rollout_size}, which is not divisible by {batch_size}"
        )
    return batch_size


def build_train_command(args: argparse.Namespace) -> list[str]:
    experiment_name = args.experiment_name or (
        f"soma23_bones_seed_mini_fsq_{args.algorithm}_seed{args.seed}"
    )
    batch_size = resolved_batch_size(args)
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
        str(batch_size),
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
        if args.sac_no_fsq and args.sac_ppo_actor_checkpoint is not None:
            raise ValueError(
                "--sac-no-fsq cannot be combined with --sac-ppo-actor-checkpoint"
            )
        overrides.extend(
            [
                f"agent.target_entropy_scale={args.target_entropy_scale}",
                f"agent.replay_buffer_size={args.replay_buffer_size}",
                f"agent.replay_memory_limit_gib={args.replay_memory_limit_gib}",
                f"agent.replay_warmup_transitions={args.replay_warmup_transitions}",
                f"agent.actor_start_training_epoch={args.sac_actor_start_epoch}",
                f"agent.num_mini_batches={args.sac_num_mini_batches}",
                f"agent.policy_frequency={args.sac_policy_frequency}",
                f"agent.actor_learning_rate={args.sac_actor_learning_rate}",
                f"agent.critic_learning_rate={args.sac_critic_learning_rate}",
                f"agent.diagnostic_batch_size={args.sac_diagnostic_batch_size}",
                f"agent.diagnostic_every={args.sac_diagnostic_every}",
                f"agent.model.min_log_std={args.sac_min_log_std}",
                f"agent.model.max_log_std={args.sac_max_log_std}",
                f"agent.model.actor_trust_region_coef={args.sac_actor_trust_region_coef}",
                f"agent.model.actor_reference_tau={args.sac_actor_reference_tau}",
            ]
        )
        if args.sac_fixed_std is not None:
            overrides.extend(
                [
                    f"agent.model.initial_std={args.sac_fixed_std}",
                    "agent.model.learn_std=false",
                ]
            )
        if args.sac_no_fsq:
            overrides.append("agent.model.use_fsq=false")
        if args.sac_ppo_actor_checkpoint is not None:
            checkpoint = args.sac_ppo_actor_checkpoint.expanduser().resolve()
            overrides.extend(
                [
                    "agent.model.ppo_compatible_normalization=true",
                    f"agent.model.ppo_actor_checkpoint={checkpoint}",
                ]
            )
        if args.sac_freeze_normalization:
            overrides.append("agent.model.freeze_normalization=true")
        if args.sac_tracking_termination_threshold is not None:
            overrides.append(
                "env.termination_components.tracking_error.static_params.threshold="
                f"{args.sac_tracking_termination_threshold}"
            )
    overrides.append(f"agent.evaluator.eval_metrics_every={args.eval_every}")
    if args.train_motion_id is not None:
        excluded = [
            motion_id
            for motion_id in range(MOTION_COUNT)
            if motion_id != args.train_motion_id
        ]
        overrides.extend(
            [
                f"env.motion_manager.exclude_motion_ids={excluded}",
                f"agent.evaluator.evaluation_motion_ids=[{args.train_motion_id}]",
            ]
        )
    if args.fixed_starts:
        overrides.append("env.motion_manager.init_start_prob=1.0")
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


def parse_evaluation_output(output: str, checkpoint: Path) -> dict:
    metrics: dict[str, float] = {}
    score = None
    num_evaluated = None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("eval/") and ":" in stripped:
            key, value = stripped.split(":", 1)
            metrics[key] = float(value.strip())
        elif stripped.startswith("Items Evaluated:"):
            num_evaluated = int(stripped.split(":", 1)[1])
        elif stripped.startswith("Overall Score:"):
            score = float(stripped.split(":", 1)[1])

    if num_evaluated is None:
        num_evaluated = int(metrics.get("eval/num_evaluated", 0))
    if num_evaluated != 61:
        raise ValueError(f"expected all 61 fixed-order motions, evaluated {num_evaluated}")

    import torch

    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metrics.update(
        {
            "score": score,
            "num_evaluated": num_evaluated,
            "fixed_order_motion_ids": list(range(num_evaluated)),
            "epoch": int(checkpoint_state["epoch"]),
            "step_count": int(checkpoint_state["step_count"]),
        }
    )
    return metrics


def run_evaluation(args: argparse.Namespace) -> int:
    command = build_eval_command(args)
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    print(completed.stdout, end="")
    print(completed.stderr, end="", file=sys.stderr)
    completed.check_returncode()

    checkpoint = args.checkpoint.expanduser().resolve()
    output_path = args.output or checkpoint.parent / "evaluation_final.json"
    result = parse_evaluation_output(completed.stdout, checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {output_path}")
    return 0


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
    train.add_argument("--batch-size", type=int)
    train.add_argument("--training-steps", type=int, default=DEFAULT_TRAINING_STEPS)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--experiment-name")
    train.add_argument("--use-wandb", action="store_true")
    train.add_argument("--target-entropy-scale", type=float, default=DEFAULT_TARGET_ENTROPY_SCALE)
    train.add_argument("--replay-buffer-size", type=int, default=262_144)
    train.add_argument("--replay-memory-limit-gib", type=float, default=9.0)
    train.add_argument("--replay-warmup-transitions", type=int, default=0)
    train.add_argument("--sac-actor-start-epoch", type=int, default=0)
    train.add_argument("--sac-num-mini-batches", type=int, default=13)
    train.add_argument("--sac-policy-frequency", type=int, default=1)
    train.add_argument("--sac-actor-learning-rate", type=float, default=2e-4)
    train.add_argument("--sac-critic-learning-rate", type=float, default=2e-4)
    train.add_argument("--sac-fixed-std", type=float)
    train.add_argument("--sac-no-fsq", action="store_true")
    train.add_argument("--sac-ppo-actor-checkpoint", type=Path)
    train.add_argument("--sac-freeze-normalization", action="store_true")
    train.add_argument("--sac-tracking-termination-threshold", type=float)
    train.add_argument("--sac-min-log-std", type=float, default=-20.0)
    train.add_argument("--sac-max-log-std", type=float, default=2.0)
    train.add_argument("--sac-actor-trust-region-coef", type=float, default=0.0)
    train.add_argument("--sac-actor-reference-tau", type=float, default=0.01)
    train.add_argument("--sac-diagnostic-batch-size", type=int, default=1024)
    train.add_argument("--sac-diagnostic-every", type=int, default=10)
    train.add_argument("--eval-every", type=int, default=200)
    train.add_argument("--train-motion-id", type=int, choices=range(MOTION_COUNT))
    train.add_argument("--fixed-starts", action="store_true")
    train.add_argument("--overrides", nargs="*", default=[])
    train.add_argument("--print-command", action="store_true")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--motion-file", type=Path, default=default_motion_path())
    evaluate.add_argument("--output", type=Path)
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
        if args.print_command:
            print(shlex.join(build_eval_command(args)))
            return 0
        return run_evaluation(args)

    if args.print_command:
        print(shlex.join(command))
        return 0
    os.execvpe(sys.executable, command, os.environ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
