#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TASKS=${TASKS:-"h1touch-door-v0"}
SEEDS=${SEEDS:-"0"}
TRAIN_STEPS=${TRAIN_STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-2000}
NUM_ENVS=${NUM_ENVS:-1}
NOISES=${NOISES:-"0.0"}
DROPS=${DROPS:-"0.0"}
RUNS_DIR=${RUNS_DIR:-outputs/runs_debug}
RESULTS_CSV=${RESULTS_CSV:-outputs/results/results_debug.csv}
ANALYSIS_CSV=${ANALYSIS_CSV:-outputs/results/contact_analysis_debug.csv}
FIGS_DIR=${FIGS_DIR:-outputs/figs_debug}
DRY_RUN=${DRY_RUN:-0}

echo "Smoke test configuration"
echo "TASKS=$TASKS"
echo "SEEDS=$SEEDS"
echo "TRAIN_STEPS=$TRAIN_STEPS"
echo "EVAL_STEPS=$EVAL_STEPS"
echo "NUM_ENVS=$NUM_ENVS"
echo "Expected budget: about 30-60 minutes on a GPU, shorter in DRY_RUN=1"

TASKS="$TASKS" \
SEEDS="$SEEDS" \
TRAIN_STEPS="$TRAIN_STEPS" \
EVAL_STEPS="$EVAL_STEPS" \
NUM_ENVS="$NUM_ENVS" \
NOISES="$NOISES" \
DROPS="$DROPS" \
RUNS_DIR="$RUNS_DIR" \
RESULTS_CSV="$RESULTS_CSV" \
ANALYSIS_CSV="$ANALYSIS_CSV" \
FIGS_DIR="$FIGS_DIR" \
DRY_RUN="$DRY_RUN" \
bash ./run_all.sh
