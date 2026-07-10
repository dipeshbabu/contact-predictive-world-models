#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TASKS=${TASKS:-"h1touch-push-v0 h1touch-door-v0 h1touch-cabinet-v0 h1touch-insert_small-v0"}
VARIANTS=${VARIANTS:-"base frontier frontier_bct_native frontier_bct_spatial frontier_rgb_bct_spatial"}
SEEDS=${SEEDS:-"0 1 2"}
BUDGETS=${BUDGETS:-"250000 500000 1000000 2000000"}
MODEL_CONFIGS=${MODEL_CONFIGS:-"humanoid_benchmark humanoid_benchmark,large"}
DRY_RUN=${DRY_RUN:-0}

for budget in $BUDGETS; do
  for config in $MODEL_CONFIGS; do
    tag="$(echo "${config}" | tr ',' '_')_${budget}"
    DREAMER_TASKS="$TASKS" \
    DREAMER_VARIANTS="$VARIANTS" \
    SEEDS="$SEEDS" \
    TRAIN_STEPS="$budget" \
    DREAMER_CONFIGS="$config" \
    RUN_PPO=0 \
    RUN_DYNAMICS=0 \
    RUNS_DIR="outputs/runs_scaling/${tag}" \
    RESULTS_CSV="outputs/results/results_scaling_${tag}.csv" \
    DRY_RUN="$DRY_RUN" \
    bash ./run_all.sh
  done
done
