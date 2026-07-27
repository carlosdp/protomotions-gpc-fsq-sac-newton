#!/usr/bin/env python3
"""Print the latest TensorBoard scalars and evaluation from a training result."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from tensorboardX.proto.event_pb2 import Event


def read_scalars(event_path: Path) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    with event_path.open("rb") as stream:
        while raw_length := stream.read(8):
            if len(raw_length) != 8:
                break
            length = struct.unpack("<Q", raw_length)[0]
            stream.read(4)
            payload = stream.read(length)
            stream.read(4)
            event = Event.FromString(payload)
            if not event.HasField("summary"):
                continue
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    series.setdefault(value.tag, []).append(
                        (event.step, value.simple_value)
                    )
    return series


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    event_paths = sorted(args.result_dir.glob("lightning_logs/version_*/events.out.*"))
    if not event_paths:
        raise FileNotFoundError(f"no TensorBoard event file beneath {args.result_dir}")
    series = read_scalars(event_paths[-1])
    interesting_prefixes = (
        "env/",
        "eval/",
        "fsq/",
        "grad/",
        "info/",
        "losses/",
        "optimization/",
        "policy/",
        "replay/",
        "rewards/",
        "sac/",
        "sac_diag/",
        "terminations/",
        "times/training_minutes",
    )
    latest = {
        key: {"step": values[-1][0], "value": values[-1][1]}
        for key, values in sorted(series.items())
        if key.startswith(interesting_prefixes)
    }
    evaluation_path = args.result_dir / "evaluation_final.json"
    output = {
        "result_dir": str(args.result_dir),
        "latest": latest,
        "evaluation": (
            json.loads(evaluation_path.read_text())
            if evaluation_path.exists()
            else None
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
