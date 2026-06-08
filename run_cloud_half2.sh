#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name --format=csv,noheader || true
fi

HALF_TASKS=${HALF_TASKS:-"h1touch-run-v0 h1touch-door-v0 h1touch-insert_small-v0"}
RUNS_DIR=${RUNS_DIR:-outputs/runs_half2}
RESULTS_CSV=${RESULTS_CSV:-outputs/results/results_half2.csv}
ANALYSIS_CSV=${ANALYSIS_CSV:-outputs/results/contact_analysis_half2.csv}
FIGS_DIR=${FIGS_DIR:-outputs/figs_half2}
TACTILE_ERROR_CSV=${TACTILE_ERROR_CSV:-outputs/results/tactile_error_vs_success_half2.csv}
TACTILE_ERROR_FIG=${TACTILE_ERROR_FIG:-outputs/figs_half2/tactile_error_vs_success.png}
DYNAMICS_CSV=${DYNAMICS_CSV:-outputs/results/dynamics_summary_half2.csv}

mkdir -p logs "$RUNS_DIR" "$(dirname "$RESULTS_CSV")" "$FIGS_DIR"

echo "Half2 cloud run"
echo "DREAMER_TASKS=$HALF_TASKS"
echo "PPO_TASKS=$HALF_TASKS"
echo "RUNS_DIR=$RUNS_DIR"
echo "RESULTS_CSV=$RESULTS_CSV"

DREAMER_TASKS="$HALF_TASKS" \
PPO_TASKS="$HALF_TASKS" \
RUNS_DIR="$RUNS_DIR" \
RESULTS_CSV="$RESULTS_CSV" \
ANALYSIS_CSV="$ANALYSIS_CSV" \
FIGS_DIR="$FIGS_DIR" \
TACTILE_ERROR_CSV="$TACTILE_ERROR_CSV" \
TACTILE_ERROR_FIG="$TACTILE_ERROR_FIG" \
DYNAMICS_CSV="$DYNAMICS_CSV" \
JAX_PLATFORM="${JAX_PLATFORM:-gpu}" \
bash ./run_all.sh
