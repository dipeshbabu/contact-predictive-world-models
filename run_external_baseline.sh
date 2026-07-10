#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BASELINE_BACKEND=${BASELINE_BACKEND:-dreamer}
BASELINE_COMMAND_TEMPLATE=${BASELINE_COMMAND_TEMPLATE:-}
BASELINE_VARIANTS=${BASELINE_VARIANTS:-"external_world_model"}
DREAMER_TASKS=${DREAMER_TASKS:-"h1touch-push-v0 h1touch-door-v0 h1touch-cabinet-v0 h1touch-insert_small-v0"}
SEEDS=${SEEDS:-"0 1 2"}
TRAIN_STEPS=${TRAIN_STEPS:-2000000}
RUNS_DIR=${RUNS_DIR:-outputs/runs_external_baselines}
DRY_RUN=${DRY_RUN:-0}

for env in $DREAMER_TASKS; do
  for variant in $BASELINE_VARIANTS; do
    for seed in $SEEDS; do
      args=(
        --backend "$BASELINE_BACKEND"
        --env "$env"
        --variant "$variant"
        --seed "$seed"
        --steps "$TRAIN_STEPS"
        --logdir "${RUNS_DIR}/${env}_${variant}_s${seed}"
      )
      if [[ -n "$BASELINE_COMMAND_TEMPLATE" ]]; then
        args+=(--command_template "$BASELINE_COMMAND_TEMPLATE")
      fi
      if [[ "$DRY_RUN" == "1" ]]; then
        args+=(--dry_run)
      fi
      python cpwm/baseline_adapter.py "${args[@]}"
    done
  done
done
