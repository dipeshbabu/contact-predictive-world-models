#!/usr/bin/env bash
set -u
set -o pipefail

FAILS=0

TASKS=${TASKS:-"h1touch-push-v0 h1touch-door-v0 h1touch-cabinet-v0 h1touch-insert_small-v0"}
SEEDS=${SEEDS:-"0 1 2"}

TRAIN_STEPS=${TRAIN_STEPS:-2000000}
EVAL_STEPS=${EVAL_STEPS:-20000}
NUM_ENVS=${NUM_ENVS:-4}

AUX_ON=${AUX_ON:-0.1}
AUX_OFF=${AUX_OFF:-0.0}

NOISES=${NOISES:-"0.0 0.01 0.02 0.05"}
DROPS=${DROPS:-"0.0 0.2 0.4 0.6"}
DRY_RUN=${DRY_RUN:-0}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUNS_DIR=${RUNS_DIR:-outputs/runs}
RESULTS_CSV=${RESULTS_CSV:-outputs/results/results.csv}
ANALYSIS_CSV=${ANALYSIS_CSV:-outputs/results/contact_analysis.csv}
FIGS_DIR=${FIGS_DIR:-outputs/figs}

mkdir -p "$RUNS_DIR" "$(dirname "$RESULTS_CSV")" "$FIGS_DIR"

echo "Root: $ROOT_DIR"
echo "TASKS=$TASKS"
echo "SEEDS=$SEEDS"
echo "TRAIN_STEPS=$TRAIN_STEPS"
echo "EVAL_STEPS=$EVAL_STEPS"
echo "NUM_ENVS=$NUM_ENVS"
echo "DRY_RUN=$DRY_RUN"

train_one () {
  local env="$1"
  local variant="$2"
  local seed="$3"
  local aux_weight="$4"
  local logdir="${RUNS_DIR}/${env}_${variant}_s${seed}"
  local args=(
    --env "${env}"
    --seed "${seed}"
    --steps "${TRAIN_STEPS}"
    --num_envs "${NUM_ENVS}"
    --tactile_aux_weight "${aux_weight}"
    --logdir "${logdir}"
  )

  if [[ "$DRY_RUN" != "1" && -f "${logdir}/checkpoint.ckpt" ]]; then
    echo "[SKIP TRAIN] ${logdir}"
    return 0
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry_run)
  fi

  echo "[TRAIN] env=${env} variant=${variant} seed=${seed} aux=${aux_weight}"
  python cpwm/train_dreamer.py "${args[@]}"
}

eval_one () {
  local run_dir="$1"
  local noise="$2"
  local drop="$3"
  local args=(
    --run_dir "${run_dir}"
    --steps "${EVAL_STEPS}"
    --noise "${noise}"
    --tactile_dropout "${drop}"
    --results_csv "${RESULTS_CSV}"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    mkdir -p "${run_dir}"
    args+=(--dry_run)
  fi

  echo "[EVAL] ${run_dir} noise=${noise} drop=${drop}"
  python cpwm/eval_dreamer.py "${args[@]}"
}

for env in $TASKS; do
  for seed in $SEEDS; do
    train_one "$env" "base" "$seed" "$AUX_OFF" || { echo "[TRAIN FAIL] $env base s$seed"; FAILS=$((FAILS+1)); }
    train_one "$env" "aux"  "$seed" "$AUX_ON"  || { echo "[TRAIN FAIL] $env aux s$seed"; FAILS=$((FAILS+1)); }
  done
done

for env in $TASKS; do
  for seed in $SEEDS; do
    for variant in base aux; do
      run_dir="${RUNS_DIR}/${env}_${variant}_s${seed}"
      if [[ "$DRY_RUN" != "1" && ! -d "${run_dir}" ]]; then
        echo "[SKIP EVAL] missing run dir ${run_dir}"
        FAILS=$((FAILS+1))
        continue
      fi

      eval_one "$run_dir" "0.0" "0.0" || { echo "[EVAL FAIL] $run_dir clean"; FAILS=$((FAILS+1)); }

      for noise in $NOISES; do
        for drop in $DROPS; do
          [[ "$noise" == "0.0" && "$drop" == "0.0" ]] && continue
          eval_one "$run_dir" "$noise" "$drop" || { echo "[EVAL FAIL] $run_dir noise=$noise drop=$drop"; FAILS=$((FAILS+1)); }
        done
      done
    done
  done
done

if [[ "$DRY_RUN" != "1" ]]; then
  python cpwm/plot_results.py \
    --csv "${RESULTS_CSV}" \
    --outdir "${FIGS_DIR}" || { echo "[PLOT FAIL]"; FAILS=$((FAILS+1)); }

  python cpwm/analysis_contact_probe.py \
    --csv "${RESULTS_CSV}" \
    --out "${ANALYSIS_CSV}" || { echo "[ANALYSIS FAIL]"; FAILS=$((FAILS+1)); }
else
  echo "[DRY RUN] skipping plot and analysis generation"
fi

echo "Done. FAILS=$FAILS"
if [[ "$DRY_RUN" != "1" ]]; then
  ls -lh "${RESULTS_CSV}" || true
  ls -lh "${ANALYSIS_CSV}" || true
  ls -lh "${FIGS_DIR}" || true
else
  echo "[DRY RUN] result files are not created"
fi

exit 0
