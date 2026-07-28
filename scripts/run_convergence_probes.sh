#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

COMMON_ARGS=(
  --num-envs 512
  --training-steps 3072000
  --train-motion-id 3
  --sac-fixed-std 0.5
  --replay-warmup-transitions 262144
  --sac-actor-start-epoch 100
  --sac-num-mini-batches 16
  --sac-policy-frequency 2
  --sac-actor-learning-rate 0.0001
  --sac-critic-learning-rate 0.0001
  --sac-conservative-q-coef 0.5
  --sac-conservative-q-batch-size 8192
  --sac-actor-trust-region-coef 1.0
  --sac-actor-reference-tau 0.005
  --sac-action-bounds-from-motion
  --sac-action-bounds-train-motion-only
  --sac-action-bounds-margin 0.02
  --no-sac-action-bounds-symmetric
  --sac-tracking-termination-threshold 2.0
  --eval-every 25
  --use-wandb
)

OVERRIDES=(
  agent.auto_alpha=false
  agent.initial_alpha=0.001
  agent.save_last_checkpoint_every=25
  agent.save_epoch_checkpoint_every=1000
  agent.save_inference_checkpoint=false
)

run_seed() {
  local seed="$1"
  local name="soma23_motion3_fsq_sac_fresh_random_seed${seed}_250"
  PYTHONUNBUFFERED=1 uv run gpc-fsq train sac \
    "${COMMON_ARGS[@]}" \
    --seed "$seed" \
    --experiment-name "$name" \
    --overrides "${OVERRIDES[@]}" \
    2>&1 | tee "logs/${name}.log"
}

for seed in ${SEEDS:-0}; do
  run_seed "$seed"
done
