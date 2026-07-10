#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DREAMER_TASKS=${DREAMER_TASKS:-"h1touch-walk-v0 h1touch-run-v0 h1touch-push-v0 h1touch-door-v0 h1touch-cabinet-v0 h1touch-insert_small-v0"}
DREAMER_VARIANTS=${DREAMER_VARIANTS:-"frontier_bct_spatial frontier_rgb_bct_spatial"}
SEEDS=${SEEDS:-"0 1 2"}
TRAIN_STEPS=${TRAIN_STEPS:-2000000}
NUM_ENVS=${NUM_ENVS:-4}
DREAMER_CONFIGS=${DREAMER_CONFIGS:-"humanoid_benchmark,large"}
RUNS_DIR=${RUNS_DIR:-outputs/runs_foundation_pretrain}
RESULTS_CSV=${RESULTS_CSV:-outputs/results/results_foundation_pretrain.csv}
DRY_RUN=${DRY_RUN:-0}

DREAMER_TASKS="$DREAMER_TASKS" \
DREAMER_VARIANTS="$DREAMER_VARIANTS" \
SEEDS="$SEEDS" \
TRAIN_STEPS="$TRAIN_STEPS" \
NUM_ENVS="$NUM_ENVS" \
DREAMER_CONFIGS="$DREAMER_CONFIGS" \
RUNS_DIR="$RUNS_DIR" \
RESULTS_CSV="$RESULTS_CSV" \
RUN_PPO=0 \
RUN_EVAL=0 \
RUN_DYNAMICS=0 \
PRETRAIN_WORLD_MODEL_ONLY=1 \
DRY_RUN="$DRY_RUN" \
bash ./run_all.sh
