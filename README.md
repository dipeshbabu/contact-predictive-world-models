# Contact Predictive World Models

Contact-predictive multimodal world model reinforcement learning for partially observable humanoid manipulation.

## Overview

This repo studies whether a contact-predictive auxiliary objective improves tactile world-model learning on H1Touch locomotion and manipulation tasks.

Core comparison:
- PPO proprio-only baseline
- Dreamer tactile baseline
- Dreamer tactile + contact-predictive auxiliary loss

Active tasks:
- `h1touch-walk-v0`
- `h1touch-run-v0`
- `h1touch-push-v0`
- `h1touch-door-v0`
- `h1touch-cabinet-v0`
- `h1touch-insert_small-v0`

Robustness evaluation:
- proprioceptive noise
- tactile dropout
- mild dynamics variation via mass scaling
- mild dynamics variation via friction scaling

## Repo Layout

```text
.
 cpwm/
    train_dreamer.py
    eval_dreamer.py
    train_ppo.py
    eval_ppo.py
    plot_results.py
    analysis_contact_probe.py
    analysis_tactile_error.py
    analysis_dynamics_summary.py
 embodied/
 humanoid_bench/
 outputs/
 setup/
    setup_env.sh
 run_all.sh
 run_debug.sh
 pyproject.toml
 requirements.txt
```

## Setup

Create the environment with `uv`:

```bash
uv venv --seed --python 3.11
source .venv/bin/activate
```

CPU install:

```bash
bash setup/setup_env.sh cpu
```

CUDA 12 install:

```bash
bash setup/setup_env.sh cuda12
```

The setup script:
- requires Python `3.11.x`
- uses `uv pip` by default
- installs pinned dependencies from `requirements.txt`
- installs the repo in editable mode via `pyproject.toml`

If `uv` is unavailable, you can use the slower pip fallback:

```bash
CPWM_INSTALLER=pip bash setup/setup_env.sh cpu
```

Headless rendering for remote Linux machines:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## Smoke Test

`run_debug.sh` is the bounded smoke test. It is the first script you should run after setup.

Default debug run:
- Dreamer tasks: `h1touch-walk-v0` and `h1touch-door-v0`
- PPO tasks: `h1touch-walk-v0` and `h1touch-door-v0`
- seed: `0`
- variants: Dreamer `base` and `aux`, plus PPO proprio-only
- Dreamer train steps: `2000`
- PPO train steps: `1000`
- Dreamer eval steps: `200`
- PPO eval episodes: `1`
- sensory checks: clean plus one noisy/dropout setting
- dynamics checks: disabled by default
- expected wall-clock: a short smoke test on a single GPU

Dry-run command wiring check:

```bash
DRY_RUN=1 bash run_debug.sh
```

Real smoke test:

```bash
bash run_debug.sh
```

Useful debug overrides:

```bash
DREAMER_TASKS="h1touch-push-v0" PPO_TASKS="h1touch-push-v0" bash run_debug.sh
TRAIN_STEPS=5000 PPO_TRAIN_STEPS=2000 EVAL_STEPS=500 PPO_EVAL_EPISODES=2 bash run_debug.sh
SEEDS="0" bash run_debug.sh
NUM_ENVS=1 bash run_debug.sh
RUN_PPO=0 bash run_debug.sh
RUN_DYNAMICS=0 bash run_debug.sh
RUN_DYNAMICS=1 bash run_debug.sh
```

Debug outputs go to:
- `outputs/runs_debug/`
- `outputs/results/results_debug.csv`
- `outputs/results/contact_analysis_debug.csv`
- `outputs/results/tactile_error_vs_success_debug.csv`
- `outputs/results/dynamics_summary_debug.csv`
- `outputs/figs_debug/`

## Full Run

Run the full proposal-complete pipeline:

```bash
bash run_all.sh
```

Two-person split wrappers:

```bash
bash run_all_half1.sh
bash run_all_half2.sh
```

Default task split:
- `run_all_half1.sh`: `h1touch-walk-v0 h1touch-push-v0 h1touch-cabinet-v0`
- `run_all_half2.sh`: `h1touch-run-v0 h1touch-door-v0 h1touch-insert_small-v0`

Each wrapper runs both:
- Dreamer base and auxiliary training plus evaluation
- PPO training plus evaluation
- isolated output paths by default, so both people can run them concurrently in the same shared workspace

```bash
bash run_all_half1.sh
bash run_all_half2.sh
```

Useful overrides:

```bash
DREAMER_TASKS="h1touch-walk-v0 h1touch-door-v0" PPO_TASKS="h1touch-walk-v0 h1touch-door-v0" bash run_all.sh
SEEDS="0 1" bash run_all.sh
TRAIN_STEPS=500000 PPO_TRAIN_STEPS=200000 EVAL_STEPS=5000 PPO_EVAL_EPISODES=5 bash run_all.sh
NUM_ENVS=2 bash run_all.sh
RUN_PPO=0 bash run_all.sh
RUN_DYNAMICS=0 bash run_all.sh
DRY_RUN=1 bash run_all.sh
```

Main outputs:
- `outputs/results/results.csv`
- `outputs/results/contact_analysis.csv`
- `outputs/results/tactile_error_vs_success.csv`
- `outputs/results/dynamics_summary.csv`
- `outputs/figs/`

Default full-run coverage:
- Dreamer base and auxiliary models on all 6 tasks
- sensory sweeps across `NOISES x DROPS`
- dynamics sweeps across `MASS_SCALES x FRICTION_SCALES` when `RUN_DYNAMICS=1`
- PPO proprio-only baseline on all 6 tasks when `RUN_PPO=1`

## Manual Commands

Train baseline:

```bash
python cpwm/train_dreamer.py \
  --env h1touch-door-v0 \
  --seed 0 \
  --steps 2000000 \
  --num_envs 4 \
  --tactile_aux_weight 0.0 \
  --logdir outputs/runs/h1touch-door-v0_base_s0
```

Train auxiliary:

```bash
python cpwm/train_dreamer.py \
  --env h1touch-door-v0 \
  --seed 0 \
  --steps 2000000 \
  --num_envs 4 \
  --tactile_aux_weight 0.1 \
  --logdir outputs/runs/h1touch-door-v0_aux_s0
```

Evaluate a checkpoint:

```bash
python cpwm/eval_dreamer.py \
  --run_dir outputs/runs/h1touch-door-v0_aux_s0 \
  --steps 20000 \
  --noise 0.0 \
  --tactile_dropout 0.0 \
  --mass_scale 1.0 \
  --friction_scale 1.0
```

Evaluate sensory robustness:

```bash
python cpwm/eval_dreamer.py \
  --run_dir outputs/runs/h1touch-door-v0_aux_s0 \
  --steps 20000 \
  --noise 0.02 \
  --tactile_dropout 0.2 \
  --mass_scale 1.0 \
  --friction_scale 1.0
```

Evaluate dynamics robustness:

```bash
python cpwm/eval_dreamer.py \
  --run_dir outputs/runs/h1touch-door-v0_aux_s0 \
  --steps 20000 \
  --noise 0.0 \
  --tactile_dropout 0.0 \
  --mass_scale 1.1 \
  --friction_scale 0.8
```

Train PPO baseline:

```bash
python cpwm/train_ppo.py \
  --env h1touch-walk-v0 \
  --seed 0 \
  --steps 1000000 \
  --logdir outputs/runs/h1touch-walk-v0_ppo_proprio_s0
```

Evaluate PPO baseline:

```bash
python cpwm/eval_ppo.py \
  --run_dir outputs/runs/h1touch-walk-v0_ppo_proprio_s0 \
  --env h1touch-walk-v0 \
  --seed 0 \
  --episodes 20 \
  --noise 0.02 \
  --tactile_dropout 0.2 \
  --mass_scale 1.0 \
  --friction_scale 1.0
```

Generate plots:

```bash
python cpwm/plot_results.py \
  --csv outputs/results/results.csv \
  --outdir outputs/figs
```

Generate summary analysis:

```bash
python cpwm/analysis_contact_probe.py \
  --csv outputs/results/results.csv \
  --out outputs/results/contact_analysis.csv
```

Generate tactile prediction error analysis:

```bash
python cpwm/analysis_tactile_error.py \
  --runs_dir outputs/runs \
  --results_csv outputs/results/results.csv \
  --out_csv outputs/results/tactile_error_vs_success.csv \
  --out_fig outputs/figs/tactile_error_vs_success.png
```

Generate dynamics summary:

```bash
python cpwm/analysis_dynamics_summary.py \
  --csv outputs/results/results.csv \
  --out outputs/results/dynamics_summary.csv
```

## Notes

- `run_all.sh` is the main proposal-complete experiment driver at the repo root.
- `run_debug.sh` is the bounded smoke test and should be used before a full run.
- `embodied/` contains the Dreamer runtime used by train and eval.
- `humanoid_bench/` contains the active H1Touch tasks, wrappers, and assets required by this repo.
- `requirements.txt` is the single dependency source of truth.
- [pyproject.toml](/C:/Users/dipes/Documents/contact-predictive-world-models/pyproject.toml) reads dependencies from [requirements.txt](/C:/Users/dipes/Documents/contact-predictive-world-models/requirements.txt).

## Acknowledgments

This repo builds on and adapts code from:

- Humanoid Bench: https://github.com/carlosferrazza/humanoid-bench
- DreamerV3: https://github.com/danijar/dreamerv3
- `jaxrl_m`: https://github.com/dibyaghosh/jaxrl_m/tree/main

This codebase contains some files adapted from those upstream projects.
