#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PPO_CHECKPOINT="${1:?usage: scripts/run_converged_sac.sh PPO_CHECKPOINT [MOTION_FILE]}"
MOTION_FILE="${2:-data/soma23_bones_seed_mini.pt}"
NUM_ENVS="${NUM_ENVS:-512}"
ITERATIONS="${ITERATIONS:-500}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
TRAINING_STEPS=$((NUM_ENVS * ROLLOUT_STEPS * ITERATIONS))
EXPERIMENT_NAME="${EXPERIMENT_NAME:-soma23_all61_fsq_sac_converged_seed0_${ITERATIONS}}"

uv run gpc-fsq train sac \
  --motion-file "$MOTION_FILE" \
  --num-envs "$NUM_ENVS" \
  --training-steps "$TRAINING_STEPS" \
  --seed 0 \
  --eval-every 100 \
  --experiment-name "$EXPERIMENT_NAME" \
  --use-wandb \
  --sac-fixed-std 0.055 \
  --replay-warmup-transitions 262144 \
  --sac-actor-start-epoch 100 \
  --sac-num-mini-batches 4 \
  --sac-policy-frequency 2 \
  --sac-actor-learning-rate 1e-4 \
  --sac-critic-learning-rate 1e-4 \
  --sac-actor-trust-region-coef 500 \
  --sac-actor-reference-tau 0.001 \
  --sac-ppo-actor-checkpoint "$PPO_CHECKPOINT" \
  --sac-freeze-normalization \
  --overrides agent.auto_alpha=false agent.initial_alpha=0.001
