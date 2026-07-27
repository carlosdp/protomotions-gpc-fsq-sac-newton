#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

COMMON_ARGS=(
  --num-envs 512
  --training-steps 6144000
  --seed 0
  --train-motion-id 43
  --fixed-starts
  --eval-every 100
  --use-wandb
)

run_probe() {
  local name="$1"
  shift
  PYTHONUNBUFFERED=1 uv run gpc-fsq train sac \
    "${COMMON_ARGS[@]}" \
    --experiment-name "$name" \
    "$@" 2>&1 | tee "logs/${name}.log"
}

if [[ "${SKIP_PROBE_A:-0}" != "1" ]]; then
  run_probe soma23_walk43_fsq_sac_probe_a_current_seed0_500 \
    --sac-num-mini-batches 13 \
    --sac-policy-frequency 1
fi

if [[ "${SKIP_PROBE_B:-0}" != "1" ]]; then
  run_probe soma23_walk43_fsq_sac_probe_b_fixed_std_seed0_500 \
    --sac-fixed-std 0.055 \
    --sac-num-mini-batches 13 \
    --sac-policy-frequency 1
fi

if [[ "${SKIP_PROBE_C:-0}" != "1" ]]; then
  run_probe soma23_walk43_fsq_sac_probe_c_replay_seed0_500 \
    --replay-warmup-transitions 262144 \
    --sac-num-mini-batches 2 \
    --sac-policy-frequency 2
fi

if [[ "${SKIP_PROBE_D:-0}" != "1" ]]; then
  run_probe soma23_walk43_fsq_sac_probe_d_combined_seed0_500 \
    --sac-fixed-std 0.055 \
    --replay-warmup-transitions 262144 \
    --sac-num-mini-batches 2 \
    --sac-policy-frequency 2
fi
