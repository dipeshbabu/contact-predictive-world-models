#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUNS_DIR=${RUNS_DIR:-outputs/runs_frontier}
DREAMER_CONFIGS=${DREAMER_CONFIGS:-humanoid_benchmark}
DREAMER_TASKS=${DREAMER_TASKS:-"h1touch-push-v0 h1touch-door-v0 h1touch-cabinet-v0 h1touch-insert_small-v0"}
DREAMER_VARIANTS=${DREAMER_VARIANTS:-"base frontier frontier_bct_native frontier_bct_spatial frontier_rgb_bct_spatial"}
SEEDS=${SEEDS:-"0 1 2"}
EVAL_STEPS=${EVAL_STEPS:-20000}
NOISES=${NOISES:-"0.0 0.02 0.05 0.1"}
DROPS=${DROPS:-"0.0 0.2 0.4 0.6 0.8"}
MASS_SCALES=${MASS_SCALES:-"0.8 0.9 1.0 1.1 1.2"}
FRICTION_SCALES=${FRICTION_SCALES:-"0.6 0.8 1.0 1.2 1.4"}
RESULTS_CSV=${RESULTS_CSV:-outputs/results/results_transfer.csv}
DRY_RUN=${DRY_RUN:-0}

for env in $DREAMER_TASKS; do
  for variant in $DREAMER_VARIANTS; do
    for seed in $SEEDS; do
      run_dir="${RUNS_DIR}/${env}_${variant}_s${seed}"
      for noise in $NOISES; do
        for drop in $DROPS; do
          args=(
            --run_dir "$run_dir"
            --configs "$DREAMER_CONFIGS"
            --env "$env"
            --seed "$seed"
            --steps "$EVAL_STEPS"
            --noise "$noise"
            --tactile_dropout "$drop"
            --mass_scale 1.0
            --friction_scale 1.0
            --results_csv "$RESULTS_CSV"
          )
          if [[ "$DRY_RUN" == "1" ]]; then
            mkdir -p "$run_dir"
            args+=(--dry_run)
          fi
          python cpwm/eval_dreamer.py "${args[@]}"
        done
      done
      for mass in $MASS_SCALES; do
        for fric in $FRICTION_SCALES; do
          [[ "$mass" == "1.0" && "$fric" == "1.0" ]] && continue
          args=(
            --run_dir "$run_dir"
            --configs "$DREAMER_CONFIGS"
            --env "$env"
            --seed "$seed"
            --steps "$EVAL_STEPS"
            --noise 0.0
            --tactile_dropout 0.0
            --mass_scale "$mass"
            --friction_scale "$fric"
            --results_csv "$RESULTS_CSV"
          )
          if [[ "$DRY_RUN" == "1" ]]; then
            mkdir -p "$run_dir"
            args+=(--dry_run)
          fi
          python cpwm/eval_dreamer.py "${args[@]}"
        done
      done
    done
  done
done
