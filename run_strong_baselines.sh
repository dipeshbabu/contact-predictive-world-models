#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DREAMER_TASKS=${DREAMER_TASKS:-"h1touch-push-v0 h1touch-door-v0 h1touch-cabinet-v0 h1touch-insert_small-v0"}
DREAMER_VARIANTS=${DREAMER_VARIANTS:-"base aux frontier frontier_bct frontier_rgb_bct"}
SEEDS=${SEEDS:-"0 1 2"}
TRAIN_STEPS=${TRAIN_STEPS:-2000000}
EVAL_STEPS=${EVAL_STEPS:-20000}
DREAMER_CONFIGS=${DREAMER_CONFIGS:-"humanoid_benchmark,large"}
RUN_PPO=${RUN_PPO:-1}
RUN_PPO_TACTILE=${RUN_PPO_TACTILE:-1}
RUN_DYNAMICS=${RUN_DYNAMICS:-1}
RUNS_DIR=${RUNS_DIR:-outputs/runs_strong_baselines}
RESULTS_CSV=${RESULTS_CSV:-outputs/results/results_strong_baselines.csv}

DREAMER_TASKS="$DREAMER_TASKS" \
DREAMER_VARIANTS="$DREAMER_VARIANTS" \
SEEDS="$SEEDS" \
TRAIN_STEPS="$TRAIN_STEPS" \
EVAL_STEPS="$EVAL_STEPS" \
DREAMER_CONFIGS="$DREAMER_CONFIGS" \
RUN_PPO="$RUN_PPO" \
RUN_PPO_TACTILE="$RUN_PPO_TACTILE" \
RUN_DYNAMICS="$RUN_DYNAMICS" \
RUNS_DIR="$RUNS_DIR" \
RESULTS_CSV="$RESULTS_CSV" \
bash ./run_all.sh
