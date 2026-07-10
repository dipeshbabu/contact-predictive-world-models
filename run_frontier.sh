#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Focused frontier matrix: manipulation tasks, richer contact objectives,
# masked tactile modeling, grouped tactile summaries, and optional RGB.
DREAMER_TASKS=${DREAMER_TASKS:-"h1touch-push-v0 h1touch-door-v0 h1touch-cabinet-v0 h1touch-insert_small-v0"}
PPO_TASKS=${PPO_TASKS:-"$DREAMER_TASKS"}
DREAMER_VARIANTS=${DREAMER_VARIANTS:-"base aux contact both masked tactile_group part_tokens contact_frontier frontier frontier_bct"}
SEEDS=${SEEDS:-"0 1 2"}
NOISES=${NOISES:-"0.0 0.02 0.05"}
DROPS=${DROPS:-"0.0 0.2 0.4"}
RUN_DYNAMICS=${RUN_DYNAMICS:-1}
RUN_PPO=${RUN_PPO:-1}
RUN_PPO_TACTILE=${RUN_PPO_TACTILE:-1}
RUNS_DIR=${RUNS_DIR:-outputs/runs_frontier}
RESULTS_CSV=${RESULTS_CSV:-outputs/results/results_frontier.csv}
ANALYSIS_CSV=${ANALYSIS_CSV:-outputs/results/contact_analysis_frontier.csv}
FIGS_DIR=${FIGS_DIR:-outputs/figs_frontier}
TACTILE_ERROR_CSV=${TACTILE_ERROR_CSV:-outputs/results/tactile_error_vs_success_frontier.csv}
TACTILE_ERROR_FIG=${TACTILE_ERROR_FIG:-outputs/figs_frontier/tactile_error_vs_success.png}
DYNAMICS_CSV=${DYNAMICS_CSV:-outputs/results/dynamics_summary_frontier.csv}

DREAMER_TASKS="$DREAMER_TASKS" \
PPO_TASKS="$PPO_TASKS" \
DREAMER_VARIANTS="$DREAMER_VARIANTS" \
SEEDS="$SEEDS" \
NOISES="$NOISES" \
DROPS="$DROPS" \
RUN_DYNAMICS="$RUN_DYNAMICS" \
RUN_PPO="$RUN_PPO" \
RUN_PPO_TACTILE="$RUN_PPO_TACTILE" \
RUNS_DIR="$RUNS_DIR" \
RESULTS_CSV="$RESULTS_CSV" \
ANALYSIS_CSV="$ANALYSIS_CSV" \
FIGS_DIR="$FIGS_DIR" \
TACTILE_ERROR_CSV="$TACTILE_ERROR_CSV" \
TACTILE_ERROR_FIG="$TACTILE_ERROR_FIG" \
DYNAMICS_CSV="$DYNAMICS_CSV" \
bash ./run_all.sh
