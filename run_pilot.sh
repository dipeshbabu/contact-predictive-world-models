#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Small first-pass matrix for checking whether the tactile/contact objectives
# produce a real signal before launching the expensive full sweep.
DREAMER_TASKS=${DREAMER_TASKS:-"h1touch-walk-v0 h1touch-push-v0 h1touch-door-v0"}
PPO_TASKS=${PPO_TASKS:-"$DREAMER_TASKS"}
DREAMER_VARIANTS=${DREAMER_VARIANTS:-"proprio base aux recon noact contact both contact_onset both_onset"}
SEEDS=${SEEDS:-"0 1 2"}
NOISES=${NOISES:-"0.0 0.02"}
DROPS=${DROPS:-"0.0 0.2"}
RUN_DYNAMICS=${RUN_DYNAMICS:-0}
RUN_PPO=${RUN_PPO:-1}
RUN_PPO_TACTILE=${RUN_PPO_TACTILE:-1}
RUNS_DIR=${RUNS_DIR:-outputs/runs_pilot}
RESULTS_CSV=${RESULTS_CSV:-outputs/results/results_pilot.csv}
ANALYSIS_CSV=${ANALYSIS_CSV:-outputs/results/contact_analysis_pilot.csv}
FIGS_DIR=${FIGS_DIR:-outputs/figs_pilot}
TACTILE_ERROR_CSV=${TACTILE_ERROR_CSV:-outputs/results/tactile_error_vs_success_pilot.csv}
TACTILE_ERROR_FIG=${TACTILE_ERROR_FIG:-outputs/figs_pilot/tactile_error_vs_success.png}
DYNAMICS_CSV=${DYNAMICS_CSV:-outputs/results/dynamics_summary_pilot.csv}

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
