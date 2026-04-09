#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Balanced split B:
# - 1 locomotion task
# - 2 manipulation tasks
HALF_TASKS=${HALF_TASKS:-"h1touch-run-v0 h1touch-door-v0 h1touch-insert_small-v0"}
RUNS_DIR=${RUNS_DIR:-outputs/runs_half2}
RESULTS_CSV=${RESULTS_CSV:-outputs/results/results_half2.csv}
ANALYSIS_CSV=${ANALYSIS_CSV:-outputs/results/contact_analysis_half2.csv}
FIGS_DIR=${FIGS_DIR:-outputs/figs_half2}
TACTILE_ERROR_CSV=${TACTILE_ERROR_CSV:-outputs/results/tactile_error_vs_success_half2.csv}
TACTILE_ERROR_FIG=${TACTILE_ERROR_FIG:-outputs/figs_half2/tactile_error_vs_success.png}
DYNAMICS_CSV=${DYNAMICS_CSV:-outputs/results/dynamics_summary_half2.csv}

echo "Half-run B"
echo "DREAMER_TASKS=$HALF_TASKS"
echo "PPO_TASKS=$HALF_TASKS"
echo "RUNS_DIR=$RUNS_DIR"
echo "RESULTS_CSV=$RESULTS_CSV"
echo "FIGS_DIR=$FIGS_DIR"

DREAMER_TASKS="$HALF_TASKS" \
PPO_TASKS="$HALF_TASKS" \
RUNS_DIR="$RUNS_DIR" \
RESULTS_CSV="$RESULTS_CSV" \
ANALYSIS_CSV="$ANALYSIS_CSV" \
FIGS_DIR="$FIGS_DIR" \
TACTILE_ERROR_CSV="$TACTILE_ERROR_CSV" \
TACTILE_ERROR_FIG="$TACTILE_ERROR_FIG" \
DYNAMICS_CSV="$DYNAMICS_CSV" \
bash ./run_all.sh
