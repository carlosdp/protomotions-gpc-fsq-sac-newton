# ProtoMotions SOMA23 FSQ tracker with released SAC

This is a true downstream [ProtoMotions](https://github.com/NVlabs/ProtoMotions)
project. It trains the public SOMA23 finite-scalar-quantized (FSQ) motion tracker
with the SAC implementation released with
[RSL-RL: Reinforcement Learning Library for Robotics Research](https://sabagian.github.io/sac_release_project/),
then supports a controlled comparison with ProtoMotions' existing PPO-FSQ tracker.

ProtoMotions is installed as a dependency and owns the Newton simulator, SOMA23
robot/assets, motion task, rewards, terminations, evaluation, and configuration
lifecycle. This repository does not contain a ProtoMotions checkout, copy its source,
or redistribute BONES-SEED motion data.

## Scope

This project is a small-corpus algorithm-integration comparison, not a reproduction of
GPC's corpus-scale result. Both learners use the same:

- SOMA23 robot and installed ProtoMotions assets;
- exact 61-motion `soma23_bones_seed_mini.pt` fixture;
- Newton simulator and task timing;
- current-state and future-reference observations;
- 40 scalar, 9 level FSQ representation;
- rewards, termination rules, motion sampling, seeds, and evaluation motions;
- requested environment-interaction budget.

The SAC learner is the released `rsl_rl_sac` implementation pinned to commit
`e0d243aa6d3f8a7231783b7f3cefeaec1b4a5521`. Its actor uses a state-independent
vector standard deviation, as ProtoMotions policies commonly do. SAC applies the
single action `tanh`; the SAC task disables ProtoMotions' action transform so actions
are not squashed twice. The PPO baseline emits its Gaussian action before the task's
single standard ProtoMotions `tanh`.

## Requirements

- Linux x86_64
- Python 3.11
- NVIDIA GPU/driver compatible with the locked PyTorch 2.7 CUDA 12.8 stack
- Git, Git LFS, and [uv](https://docs.astral.sh/uv/)
- Weights & Biases credentials for online logging

Install from this repository:

```bash
git clone https://github.com/carlosdp/protomotions-gpc-fsq-sac-newton.git
cd protomotions-gpc-fsq-sac-newton
uv sync --locked
uv run protomotions info --json
```

The lock pins ProtoMotions to the package-enabled fork revision
`7fd6d2a82d2cf6953307acc92f021d68231ea89d` and the released SAC source to the
revision above.

## Motion fixture

Review the [BONES SEED license](https://bones.studio/info/seed-license), then explicitly
acknowledge it when fetching the official ProtoMotions fixture:

```bash
uv run gpc-fsq fetch-motion --accept-bones-seed-license
uv run gpc-fsq verify-motion
```

The fetch is pinned and verifies:

- byte SHA-256:
  `4189b48d5343e753c79081d27274186e091be4b451b1b7e326296806b6865502`;
- size: `26,719,897` bytes;
- ordered motion count: `61`;
- ordered manifest SHA-256:
  `0ed4c8f78a061154467cf583c6bb677a63c80b4a78f7f3b1c6b365a5606d18ec`.

Data is written beneath ignored `data/`; it is never committed by this project.

## Train SAC

Authenticate W&B without putting a secret in the repository:

```bash
uv run wandb login
```

Start SAC:

```bash
uv run gpc-fsq train sac \
  --num-envs 512 \
  --training-steps 25165824 \
  --seed 0 \
  --use-wandb
```

The baseline target-entropy scale is `0.167`; `0.5` is intentionally reserved for a
later ablation. The released settings retained here include 24 collection steps,
5-step targets, discount `0.97`, target smoothing `0.003`, 200 SAC minibatches of
8,192 samples, actor/critic learning rates `2e-4`, and temperature learning rate
`2e-5`.

Before allocating replay, the agent writes `replay_profile.json` with actual observation
dimensions, byte cost, requested/effective capacity, and GPU allocation. The default
requests 262,144 transitions under a 9 GiB ceiling. Checkpoints contain models,
optimizers, learned temperature, counters, environment state, and normalization state,
but deliberately do **not** snapshot replay contents yet. A resume therefore begins with
an empty replay buffer; `replay_contents_saved=false` is recorded in every checkpoint.

W&B logging uses ProtoMotions' `physical_animation` project. Run names include the
algorithm and seed, and metrics include twin critic losses, actor loss, alpha loss and
value, reward, episode length, replay occupancy, throughput, and FSQ utilization.

## Matched PPO baseline

Use the exact same interaction count and seed:

```bash
uv run gpc-fsq train ppo \
  --num-envs 512 \
  --training-steps 25165824 \
  --seed 0 \
  --use-wandb
```

PPO retains the public ProtoMotions FSQ architecture and optimizer configuration. Equal
environment interactions are the comparison invariant; SAC optimizer updates and wall
clock are logged separately because off-policy and on-policy update counts are not
equivalent.

## Evaluate and compare

Training performs a deterministic full evaluation in fixed motion-ID order and writes
`evaluation_final.json`. To re-run a checkpoint evaluation:

```bash
uv run gpc-fsq evaluate \
  --checkpoint results/soma23_bones_seed_mini_fsq_sac_seed0/last.ckpt
```

Compare completed results:

```bash
uv run gpc-fsq compare \
  --sac results/soma23_bones_seed_mini_fsq_sac_seed0 \
  --ppo results/soma23_bones_seed_mini_fsq_ppo_seed0 \
  --output results/soma23_mini_seed0_comparison.json
```

The comparison refuses unequal step counts or anything other than 61 evaluated motions.
For a more defensible result, repeat both algorithms at seeds 1 and 2; do not pool a run
with a different corpus, simulator, action mapping, or interaction budget.

## Acceptance checks

```bash
uv run ruff check .
uv run pytest
uv build
uv run gpc-fsq train sac --print-command
uv run gpc-fsq train ppo --print-command
```

A successful integration requires more than configuration construction: SAC must collect
Newton transitions, populate the released replay buffer, execute finite twin-Q, actor,
and temperature updates, save a reloadable checkpoint, and complete a fixed-order
61-motion evaluation. The PPO comparison must begin from a fresh run and finish with the
same environment-interaction count.

