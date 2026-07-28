#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

experiment_name="${1:-soma23_full61_fsq_sac_fresh_random_progressive_seed0_7500}"
training_steps="${TRAINING_STEPS:-92160000}"

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-soma23-mini-fsq-sac-random-init}"
export WANDB_TAGS="${WANDB_TAGS:-soma23-mini,fsq,newton,sac,random-init,no-ppo,from-scratch,no-checkpoint,progressive-six-to-all61,actor-pause25,replay-turnover}"

exec .venv/bin/gpc-fsq train sac \
  --motion-file data/soma23_bones_seed_mini.pt \
  --num-envs 512 \
  --training-steps "$training_steps" \
  --experiment-name "$experiment_name" \
  --use-wandb \
  --target-entropy-scale 0.167 \
  --replay-buffer-size 262144 \
  --replay-memory-limit-gib 9 \
  --replay-warmup-transitions 262144 \
  --sac-actor-start-epoch 25 \
  --sac-num-mini-batches 16 \
  --sac-policy-frequency 4 \
  --sac-actor-learning-rate 3e-5 \
  --sac-critic-learning-rate 1e-4 \
  --sac-conservative-q-coef 0.5 \
  --sac-conservative-q-batch-size 8192 \
  --sac-reference-residual-actions \
  --sac-reference-residual-action-scale 0.25 \
  --sac-reference-action-gain 0.75 \
  --sac-reference-action-time-offset-steps 1 \
  --sac-actor-trust-region-coef 1.0 \
  --sac-actor-reference-tau 0.01 \
  --sac-tracking-termination-threshold 2.0 \
  --eval-every 50 \
  --overrides \
  "agent.model.critic_num_motion_heads=61" \
  "agent.evaluator.progressive_seed_motion_ids=[3,9,20,30,43,45]" \
  "agent.evaluator.progressive_expand_success_rate=1.0" \
  "agent.evaluator.progressive_expand_consecutive_evals=1" \
  "agent.evaluator.progressive_actor_pause_epochs=25" \
  "agent.evaluator.continuous_error_curriculum=true" \
  "agent.evaluator.continuous_error_curriculum_start_epoch=500" \
  "agent.evaluator.continuous_error_curriculum_min_success_rate=0.45" \
  "agent.evaluator.continuous_error_curriculum_alpha=0.1" \
  "agent.evaluator.continuous_error_curriculum_min_relative_weight=0.75" \
  "agent.evaluator.continuous_error_curriculum_max_relative_weight=1.5" \
  "agent.save_epoch_checkpoint_every=10000" \
  "agent.save_last_checkpoint_every=100"
