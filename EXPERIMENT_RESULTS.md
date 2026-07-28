# Scratch SAC validation results

These results validate the random-initialization integration on the public
61-motion `soma23_bones_seed_mini.pt` fixture. They are a single-seed,
small-corpus engineering result, not a reproduction of GPC's corpus-scale
training and not evidence that SAC is more sample-efficient than PPO.

Both SAC lineages started with randomly initialized actor, twin critics, FSQ
encoder/decoder, normalization statistics, and an empty replay buffer. Their
commands contained no `--checkpoint` and did not consume PPO weights,
normalizers, replay, actions, or teacher outputs.

## Progressive 5,000-iteration run

- W&B: [modbdv61](https://wandb.ai/destroy-robots/physical_animation/runs/modbdv61)
- Seed: `0`
- Environments × rollout: `512 × 24`
- Environment interactions: `61,440,000`
- Seed curriculum motions: `3, 9, 20, 30, 43, 45`
- Expanded to all 61 motions: iteration `1,700`
- Iteration-5,000 evaluation: `49/61` (`80.33%`)
- Mean / maximum ground-tracking error: `0.225 / 4.512`
- Immediate final repeat: `47/61` (`77.05%`), mean / maximum
  `0.242 / 4.925`

## Progressive 7,500-iteration run

- W&B: [vzha84gd](https://wandb.ai/destroy-robots/physical_animation/runs/vzha84gd)
- Seed: `0`
- Environments × rollout: `512 × 24`
- Environment interactions: `92,160,000`
- Seed curriculum motions: `3, 9, 20, 30, 43, 45`
- Expanded to all 61 motions: iteration `1,600`
- Actor held for critic/replay turnover until iteration `1,625`
- Matched iteration-5,000 evaluation: `50/61` (`81.97%`), mean /
  maximum ground-tracking error `0.224 / 3.537`
- Best periodic evaluation at iteration `6,800`: `53/61` (`86.89%`),
  mean / maximum `0.178 / 2.636`
- Iteration-7,500 evaluation: `51/61` (`83.61%`), mean / maximum
  `0.183 / 2.884`
- Immediate final repeat: `50/61` (`81.97%`), mean / maximum
  `0.185 / 2.897`

The strict success metric fails a complete motion when any evaluated frame
crosses the ground-tracking threshold, so near-threshold clips can change the
aggregate pass count across immediate repeated evaluations. Preserve both the
periodic and final-repeat numbers when comparing runs.

## Interpretation

The previous equal-budget random-init SAC baseline was `0/61`; the corrected
5,000-iteration scratch SAC runs reached `49/61` and `50/61`. The matched PPO
baseline reached `51/61`. The result therefore establishes that SAC can learn
this tracker without PPO initialization. Multiple random seeds and
interaction-budget learning curves are still required to determine comparative
sample efficiency.
