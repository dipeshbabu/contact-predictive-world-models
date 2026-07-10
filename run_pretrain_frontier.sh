#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DREAMER_TASKS=${DREAMER_TASKS:-"h1touch-push-v0 h1touch-door-v0 h1touch-cabinet-v0 h1touch-insert_small-v0"}
DREAMER_VARIANTS=${DREAMER_VARIANTS:-"frontier_bct frontier_rgb_bct"}
SEEDS=${SEEDS:-"0 1 2"}
TRAIN_STEPS=${TRAIN_STEPS:-1000000}
NUM_ENVS=${NUM_ENVS:-4}
RUNS_DIR=${RUNS_DIR:-outputs/runs_pretrain_frontier}
RESULTS_CSV=${RESULTS_CSV:-outputs/results/results_pretrain_frontier.csv}
RUN_PPO=${RUN_PPO:-0}
RUN_EVAL=${RUN_EVAL:-0}
RUN_DYNAMICS=${RUN_DYNAMICS:-0}
DRY_RUN=${DRY_RUN:-0}

DREAMER_TASKS="$DREAMER_TASKS" \
DREAMER_VARIANTS="$DREAMER_VARIANTS" \
SEEDS="$SEEDS" \
TRAIN_STEPS="$TRAIN_STEPS" \
NUM_ENVS="$NUM_ENVS" \
RUNS_DIR="$RUNS_DIR" \
RESULTS_CSV="$RESULTS_CSV" \
RUN_PPO="$RUN_PPO" \
RUN_EVAL="$RUN_EVAL" \
RUN_DYNAMICS="$RUN_DYNAMICS" \
DRY_RUN="$DRY_RUN" \
PRETRAIN_WORLD_MODEL_ONLY=1 \
bash ./run_all.sh
