# Contact Predictive World Models

Contact-predictive multimodal world model reinforcement learning for partially observable humanoid manipulation.

## Overview

This repo studies whether a contact-predictive auxiliary objective improves tactile world-model learning on contact-rich H1Touch manipulation tasks.

Core comparison:
- Dreamer tactile baseline
- Dreamer tactile + contact-predictive auxiliary loss

Active tasks:
- `h1touch-push-v0`
- `h1touch-door-v0`
- `h1touch-cabinet-v0`
- `h1touch-insert_small-v0`

Robustness evaluation:
- proprioceptive noise
- tactile dropout

## Repo Layout

```text
.
 cpwm/
    train_dreamer.py
    eval_dreamer.py
    plot_results.py
    analysis_contact_probe.py
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

Create the environment:

```bash
conda create -n cpwm python=3.11 -y
conda activate cpwm
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
- installs pinned dependencies from `requirements.txt`
- installs the repo in editable mode via `pyproject.toml`

Headless rendering for remote Linux machines:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## Smoke Test

`run_debug.sh` is the bounded smoke test. It is the first script you should run after setup.

Default debug run:
- task: `h1touch-door-v0`
- seed: `0`
- variants: `base` and `aux`
- train steps: `20000`
- eval steps: `2000`
- clean eval only
- expected wall-clock: about 30 to 60 minutes on a single GPU

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
TRAIN_STEPS=10000 EVAL_STEPS=1000 bash run_debug.sh
TASKS="h1touch-push-v0" bash run_debug.sh
SEEDS="0" bash run_debug.sh
NUM_ENVS=1 bash run_debug.sh
```

Debug outputs go to:
- `outputs/runs_debug/`
- `outputs/results/results_debug.csv`
- `outputs/results/contact_analysis_debug.csv`
- `outputs/figs_debug/`

## Full Run

Run the full pipeline:

```bash
bash run_all.sh
```

Useful overrides:

```bash
TASKS="h1touch-door-v0 h1touch-push-v0" bash run_all.sh
SEEDS="0 1" bash run_all.sh
TRAIN_STEPS=500000 EVAL_STEPS=5000 bash run_all.sh
NUM_ENVS=2 bash run_all.sh
DRY_RUN=1 bash run_all.sh
```

Main outputs:
- `outputs/results/results.csv`
- `outputs/results/contact_analysis.csv`
- `outputs/figs/`

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
  --tactile_dropout 0.0
```

Evaluate robustness:

```bash
python cpwm/eval_dreamer.py \
  --run_dir outputs/runs/h1touch-door-v0_aux_s0 \
  --steps 20000 \
  --noise 0.02 \
  --tactile_dropout 0.2
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

## Notes

- `run_all.sh` is the main experiment driver at the repo root.
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
