# Contact Predictive World Models

Contact-predictive multimodal world model reinforcement learning for partially observable humanoid manipulation.

## Overview

This repo studies whether a contact-predictive auxiliary objective improves tactile world-model learning on H1Touch locomotion and manipulation tasks.

Core comparison:
- PPO proprio-only baseline
- PPO proprio+tactile baseline
- Dreamer proprio-only baseline
- Dreamer tactile baseline
- Dreamer tactile + future tactile auxiliary loss
- Dreamer tactile + semantic future contact auxiliary loss
- Dreamer tactile + combined tactile and contact auxiliary losses
- Dreamer tactile ablations: current tactile reconstruction, multi-step future tactile prediction, and no-action future tactile prediction

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

Analysis outputs include mean, standard error, 95% confidence intervals, per-seed clean-success points, paired seed comparisons against the `base` Dreamer variant, tactile prediction error summaries, and latent contact-probe accuracy, balanced accuracy, AUROC, F1, and class balance.
Dreamer evaluation rollouts also log held-out `report_eval/contact_probe/*` metrics when enough evaluation transitions have accumulated.

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
 run_frontier.sh
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

## Pilot Experiment

`run_pilot.sh` is the recommended first research run before launching the full matrix. It uses three seeds, three tasks, clean plus one sensory corruption setting, no dynamics sweep, PPO proprio/tactile references, and the main Dreamer controls:

```bash
bash run_pilot.sh
```

Default pilot variants:
- `proprio`: Dreamer without tactile observations
- `base`: tactile Dreamer without auxiliary loss
- `aux`: one-step future tactile prediction
- `recon`: current tactile reconstruction ablation
- `noact`: future tactile prediction without action conditioning
- `contact`: one-step semantic future contact prediction
- `both`: future tactile plus future contact prediction
- `contact_onset`: one-step new-contact prediction
- `both_onset`: future tactile plus new-contact prediction

## Frontier Run

`run_frontier.sh` is the focused high-capability matrix. It keeps the regular
pipeline but adds masked tactile modeling, grouped tactile summaries, richer
semantic contact labels, class-balanced contact losses, contact-head ensembles,
and imagined-prior contact consistency:

```bash
bash run_frontier.sh
```

Default frontier variants:
- `base`: tactile Dreamer baseline
- `aux`: future tactile prediction
- `contact`: future semantic contact prediction
- `both`: tactile plus semantic contact prediction
- `masked`: masked tactile reconstruction from latent state
- `tactile_group`: grouped tactile activity prediction
- `contact_frontier`: rich class-balanced contact-onset ensemble plus imagined consistency
- `frontier`: masked tactile plus grouped tactile plus rich contact frontier objective

Optional RGB frontier run:

```bash
DREAMER_VARIANTS="frontier_rgb" bash run_frontier.sh
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

Run the full paper matrix:

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
- Dreamer base, auxiliary, reconstruction, multi-step, and no-action ablation training plus evaluation
- PPO proprio and PPO tactile training plus evaluation
- isolated output paths by default, so both people can run them concurrently in the same shared workspace

```bash
bash run_all_half1.sh
bash run_all_half2.sh
```

Useful overrides:

```bash
DREAMER_TASKS="h1touch-walk-v0 h1touch-door-v0" PPO_TASKS="h1touch-walk-v0 h1touch-door-v0" bash run_all.sh
DREAMER_VARIANTS="base aux" RUN_PPO_TACTILE=0 bash run_all.sh
SEEDS="0 1" bash run_all.sh
TRAIN_STEPS=500000 PPO_TRAIN_STEPS=200000 EVAL_STEPS=5000 PPO_EVAL_EPISODES=5 bash run_all.sh
NUM_ENVS=2 bash run_all.sh
RUN_PPO=0 bash run_all.sh
RUN_DYNAMICS=0 bash run_all.sh
DOOR_SUCCESS_PASSAGE_THRESHOLD=0.7 bash run_all.sh
CONTACT_LABEL_OVERRIDES=outputs/results/contact_label_audit_reviewed.csv bash run_all.sh
DRY_RUN=1 bash run_all.sh
```

Main outputs:
- `outputs/results/results.csv`
- `outputs/results/contact_analysis.csv`
- `outputs/results/tactile_error_vs_success.csv`
- `outputs/results/dynamics_summary.csv`
- `outputs/figs/`

Default full-run coverage:
- Dreamer `base`, `aux`, `recon`, `future3`, and `noact` variants on all 6 tasks
- sensory sweeps across `NOISES x DROPS`
- dynamics sweeps across `MASS_SCALES x FRICTION_SCALES` when `RUN_DYNAMICS=1`
- PPO proprio-only baseline on all 6 tasks when `RUN_PPO=1`
- PPO tactile baseline on all 6 tasks when `RUN_PPO=1`
- Dreamer variants are controlled through `DREAMER_VARIANTS`:
  - `proprio` or `proprio_only`: proprio-only Dreamer without tactile observations
  - `base`: tactile Dreamer without auxiliary loss
  - `aux` or `future1`: one-step future tactile prediction from latent state and action
  - `recon` or `current`: current tactile reconstruction ablation
  - `future3`: three-step future tactile prediction
  - `future5`: five-step future tactile prediction
  - `masked`: masked tactile reconstruction from world-model latent state
  - `tactile_group`: future tactile plus grouped tactile activity prediction
  - `noact` or `future1_noact`: one-step future tactile prediction without action conditioning
  - `contact` or `contact1`: one-step future contact-label prediction
  - `contact3`: three-step future contact-label prediction
  - `contact_onset` or `onset`: one-step new-contact prediction
  - `contact_change` or `change`: one-step contact-state-change prediction
  - `contact_frontier`: rich, balanced, ensemble contact-onset prediction with imagined consistency
  - `both` or `aux_contact`: combined tactile and contact prediction
  - `both_onset` or `aux_contact_onset`: combined tactile and new-contact prediction
  - `both_change` or `aux_contact_change`: combined tactile and contact-change prediction
  - `frontier`: masked tactile, grouped tactile, rich contact onset, class balancing, ensemble disagreement, and imagined consistency
  - `frontier_rgb`: `frontier` with tactile and RGB observations

## Calibration

Door and Insert success thresholds are configurable and logged as `log_success_*` metrics. Defaults are:

- Door: stand, door openness, hatch openness, and passage thresholds all `0.8`
- Insert: stand threshold `0.8`, cube-target threshold `0.9`, peg-height threshold `0.9`

Example Door calibration run:

```bash
DOOR_SUCCESS_PASSAGE_THRESHOLD=0.7 \
DREAMER_TASKS="h1touch-door-v0" \
DREAMER_VARIANTS="base aux" \
SEEDS="0" \
RUN_PPO=0 \
RUN_DYNAMICS=0 \
bash run_all.sh
```

Contact labels can be audited and hand-reviewed:

```bash
python cpwm/audit_contact_labels.py \
  --env h1touch-door-v0 \
  --steps 5000 \
  --out outputs/results/contact_label_audit_door.csv
```

Review the generated CSV, edit the `contact_*` columns, set `reviewed=1`, then pass it back into training/evaluation:

```bash
CONTACT_LABEL_OVERRIDES=outputs/results/contact_label_audit_door.csv bash run_all.sh
```

Runs log `log_contact_label_override_used` and `log_contact_label_unknown_count` so you can verify whether reviewed labels were used and whether the fallback rules still see uncategorized contacts.

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

Train proprio-only Dreamer baseline:

```bash
python cpwm/train_dreamer.py \
  --env h1touch-door-v0 \
  --seed 0 \
  --steps 2000000 \
  --num_envs 4 \
  --sensors "" \
  --tactile_aux_weight 0.0 \
  --logdir outputs/runs/h1touch-door-v0_proprio_s0
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

Train semantic contact auxiliary:

```bash
python cpwm/train_dreamer.py \
  --env h1touch-door-v0 \
  --seed 0 \
  --steps 2000000 \
  --num_envs 4 \
  --contact_aux_weight 0.1 \
  --contact_aux_mode future \
  --contact_aux_horizon 1 \
  --logdir outputs/runs/h1touch-door-v0_contact_s0
```

Train combined tactile and contact auxiliary:

```bash
python cpwm/train_dreamer.py \
  --env h1touch-door-v0 \
  --seed 0 \
  --steps 2000000 \
  --num_envs 4 \
  --tactile_aux_weight 0.1 \
  --contact_aux_weight 0.1 \
  --logdir outputs/runs/h1touch-door-v0_both_s0
```

Train contact-onset auxiliary:

```bash
python cpwm/train_dreamer.py \
  --env h1touch-door-v0 \
  --seed 0 \
  --steps 2000000 \
  --num_envs 4 \
  --contact_aux_weight 0.1 \
  --contact_aux_mode onset \
  --contact_aux_horizon 1 \
  --logdir outputs/runs/h1touch-door-v0_contact_onset_s0
```

Train masked tactile auxiliary:

```bash
python cpwm/train_dreamer.py \
  --env h1touch-door-v0 \
  --seed 0 \
  --steps 2000000 \
  --num_envs 4 \
  --tactile_aux_weight 0.1 \
  --tactile_aux_mode masked \
  --tactile_mask_prob 0.25 \
  --logdir outputs/runs/h1touch-door-v0_masked_s0
```

Train the full frontier objective:

```bash
python cpwm/train_dreamer.py \
  --env h1touch-door-v0 \
  --seed 0 \
  --steps 2000000 \
  --num_envs 4 \
  --tactile_aux_weight 0.1 \
  --tactile_aux_mode masked \
  --tactile_group_aux_weight 0.05 \
  --contact_aux_weight 0.1 \
  --contact_aux_mode onset \
  --contact_aux_balanced \
  --contact_aux_rich_labels \
  --contact_aux_ensemble 4 \
  --contact_imagine_weight 0.05 \
  --logdir outputs/runs/h1touch-door-v0_frontier_s0
```

Train ablation variants:

```bash
python cpwm/train_dreamer.py \
  --env h1touch-door-v0 \
  --seed 0 \
  --steps 2000000 \
  --num_envs 4 \
  --tactile_aux_weight 0.1 \
  --tactile_aux_mode current \
  --logdir outputs/runs/h1touch-door-v0_recon_s0

python cpwm/train_dreamer.py \
  --env h1touch-door-v0 \
  --seed 0 \
  --steps 2000000 \
  --num_envs 4 \
  --tactile_aux_weight 0.1 \
  --tactile_aux_mode future \
  --tactile_aux_horizon 3 \
  --logdir outputs/runs/h1touch-door-v0_future3_s0
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
  --num_envs 4 \
  --logdir outputs/runs/h1touch-walk-v0_ppo_proprio_s0
```

PPO uses vectorized environments and observation normalization by default, saving `vecnormalize.pkl` next to the model so evaluation can reuse the same normalization statistics.

PPO hyperparameter sweeps are available through:

```bash
DRY_RUN=1 bash run_ppo_sweep.sh
bash run_ppo_sweep.sh
```

The default sweep covers `balanced`, `large_entropy`, and `long_rollout` PPO settings across proprio and tactile variants. Results go to `outputs/results/ppo_sweep.csv`.

Train PPO tactile baseline:

```bash
python cpwm/train_ppo.py \
  --env h1touch-walk-v0 \
  --seed 0 \
  --steps 1000000 \
  --sensors tactile \
  --logdir outputs/runs/h1touch-walk-v0_ppo_tactile_s0
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

This also writes:
- `outputs/figs/clean_success_summary.csv`
- `outputs/figs/paired_success_comparisons.csv`

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
