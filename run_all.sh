#!/usr/bin/env bash
set -u
set -o pipefail

FAILS=0

LOCO_TASKS=${LOCO_TASKS:-"h1touch-walk-v0 h1touch-run-v0"}
MANIP_TASKS=${MANIP_TASKS:-"h1touch-push-v0 h1touch-door-v0 h1touch-cabinet-v0 h1touch-insert_small-v0"}
DREAMER_TASKS=${DREAMER_TASKS:-"$LOCO_TASKS $MANIP_TASKS"}
PPO_TASKS=${PPO_TASKS:-"$DREAMER_TASKS"}
SEEDS=${SEEDS:-"0 1 2"}

TRAIN_STEPS=${TRAIN_STEPS:-2000000}
PPO_TRAIN_STEPS=${PPO_TRAIN_STEPS:-1000000}
EVAL_STEPS=${EVAL_STEPS:-20000}
PPO_EVAL_EPISODES=${PPO_EVAL_EPISODES:-20}
NUM_ENVS=${NUM_ENVS:-4}

AUX_ON=${AUX_ON:-0.1}
AUX_OFF=${AUX_OFF:-0.0}

NOISES=${NOISES:-"0.0 0.01 0.02 0.05"}
DROPS=${DROPS:-"0.0 0.2 0.4 0.6"}
MASS_SCALES=${MASS_SCALES:-"1.0 0.9 1.1"}
FRICTION_SCALES=${FRICTION_SCALES:-"1.0 0.8 1.2"}
RUN_PPO=${RUN_PPO:-1}
RUN_DYNAMICS=${RUN_DYNAMICS:-1}
DRY_RUN=${DRY_RUN:-0}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUNS_DIR=${RUNS_DIR:-outputs/runs}
RESULTS_CSV=${RESULTS_CSV:-outputs/results/results.csv}
ANALYSIS_CSV=${ANALYSIS_CSV:-outputs/results/contact_analysis.csv}
FIGS_DIR=${FIGS_DIR:-outputs/figs}
TACTILE_ERROR_CSV=${TACTILE_ERROR_CSV:-outputs/results/tactile_error_vs_success.csv}
TACTILE_ERROR_FIG=${TACTILE_ERROR_FIG:-outputs/figs/tactile_error_vs_success.png}
DYNAMICS_CSV=${DYNAMICS_CSV:-outputs/results/dynamics_summary.csv}

mkdir -p "$RUNS_DIR" "$(dirname "$RESULTS_CSV")" "$FIGS_DIR"

echo "Root: $ROOT_DIR"
echo "DREAMER_TASKS=$DREAMER_TASKS"
echo "PPO_TASKS=$PPO_TASKS"
echo "SEEDS=$SEEDS"
echo "TRAIN_STEPS=$TRAIN_STEPS"
echo "PPO_TRAIN_STEPS=$PPO_TRAIN_STEPS"
echo "EVAL_STEPS=$EVAL_STEPS"
echo "PPO_EVAL_EPISODES=$PPO_EVAL_EPISODES"
echo "NUM_ENVS=$NUM_ENVS"
echo "DRY_RUN=$DRY_RUN"

train_dreamer_one () {
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

eval_dreamer_one () {
  local run_dir="$1"
  local noise="$2"
  local drop="$3"
  local mass="$4"
  local fric="$5"
  local args=(
    --run_dir "${run_dir}"
    --steps "${EVAL_STEPS}"
    --noise "${noise}"
    --tactile_dropout "${drop}"
    --mass_scale "${mass}"
    --friction_scale "${fric}"
    --results_csv "${RESULTS_CSV}"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    mkdir -p "${run_dir}"
    args+=(--dry_run)
  fi

  echo "[DREAMER EVAL] ${run_dir} noise=${noise} drop=${drop} mass=${mass} fric=${fric}"
  python cpwm/eval_dreamer.py "${args[@]}"
}

train_ppo_one () {
  local env="$1"
  local seed="$2"
  local logdir="${RUNS_DIR}/${env}_ppo_proprio_s${seed}"
  local args=(
    --env "${env}"
    --seed "${seed}"
    --steps "${PPO_TRAIN_STEPS}"
    --logdir "${logdir}"
  )

  if [[ "$DRY_RUN" != "1" && ( -f "${logdir}/ppo_model.zip" || -f "${logdir}/ppo_model" ) ]]; then
    echo "[SKIP PPO TRAIN] ${logdir}"
    return 0
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry_run)
  fi

  echo "[PPO TRAIN] env=${env} seed=${seed}"
  python cpwm/train_ppo.py "${args[@]}"
}

eval_ppo_one () {
  local run_dir="$1"
  local env="$2"
  local seed="$3"
  local noise="$4"
  local drop="$5"
  local mass="$6"
  local fric="$7"
  local args=(
    --run_dir "${run_dir}"
    --env "${env}"
    --seed "${seed}"
    --episodes "${PPO_EVAL_EPISODES}"
    --noise "${noise}"
    --tactile_dropout "${drop}"
    --mass_scale "${mass}"
    --friction_scale "${fric}"
    --results_csv "${RESULTS_CSV}"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    mkdir -p "${run_dir}"
    args+=(--dry_run)
  fi

  echo "[PPO EVAL] ${run_dir} noise=${noise} drop=${drop} mass=${mass} fric=${fric}"
  python cpwm/eval_ppo.py "${args[@]}"
}

for env in $DREAMER_TASKS; do
  for seed in $SEEDS; do
    train_dreamer_one "$env" "base" "$seed" "$AUX_OFF" || { echo "[DREAMER TRAIN FAIL] $env base s$seed"; FAILS=$((FAILS+1)); }
    train_dreamer_one "$env" "aux"  "$seed" "$AUX_ON"  || { echo "[DREAMER TRAIN FAIL] $env aux s$seed"; FAILS=$((FAILS+1)); }
  done
done

for env in $DREAMER_TASKS; do
  for seed in $SEEDS; do
    for variant in base aux; do
      run_dir="${RUNS_DIR}/${env}_${variant}_s${seed}"
      if [[ "$DRY_RUN" != "1" && ! -d "${run_dir}" ]]; then
        echo "[SKIP DREAMER EVAL] missing run dir ${run_dir}"
        FAILS=$((FAILS+1))
        continue
      fi

      for noise in $NOISES; do
        for drop in $DROPS; do
          eval_dreamer_one "$run_dir" "$noise" "$drop" "1.0" "1.0" || { echo "[DREAMER EVAL FAIL] $run_dir noise=$noise drop=$drop"; FAILS=$((FAILS+1)); }
        done
      done

      if [[ "$RUN_DYNAMICS" == "1" ]]; then
        for mass in $MASS_SCALES; do
          for fric in $FRICTION_SCALES; do
            [[ "$mass" == "1.0" && "$fric" == "1.0" ]] && continue
            eval_dreamer_one "$run_dir" "0.0" "0.0" "$mass" "$fric" || { echo "[DREAMER DYNAMICS FAIL] $run_dir mass=$mass fric=$fric"; FAILS=$((FAILS+1)); }
          done
        done
      done
    done
  done
done

if [[ "$RUN_PPO" == "1" ]]; then
  for env in $PPO_TASKS; do
    for seed in $SEEDS; do
      train_ppo_one "$env" "$seed" || { echo "[PPO TRAIN FAIL] $env s$seed"; FAILS=$((FAILS+1)); }
    done
  done

  for env in $PPO_TASKS; do
    for seed in $SEEDS; do
      run_dir="${RUNS_DIR}/${env}_ppo_proprio_s${seed}"
      if [[ "$DRY_RUN" != "1" && ! -d "${run_dir}" ]]; then
        echo "[SKIP PPO EVAL] missing run dir ${run_dir}"
        FAILS=$((FAILS+1))
        continue
      fi

      for noise in $NOISES; do
        for drop in $DROPS; do
          eval_ppo_one "$run_dir" "$env" "$seed" "$noise" "$drop" "1.0" "1.0" || { echo "[PPO SENSORY FAIL] $run_dir noise=$noise drop=$drop"; FAILS=$((FAILS+1)); }
        done
      done

      if [[ "$RUN_DYNAMICS" == "1" ]]; then
        for mass in $MASS_SCALES; do
          for fric in $FRICTION_SCALES; do
            [[ "$mass" == "1.0" && "$fric" == "1.0" ]] && continue
            eval_ppo_one "$run_dir" "$env" "$seed" "0.0" "0.0" "$mass" "$fric" || { echo "[PPO DYNAMICS FAIL] $run_dir mass=$mass fric=$fric"; FAILS=$((FAILS+1)); }
          done
        done
      fi
    done
  done
fi

if [[ "$DRY_RUN" != "1" ]]; then
  python cpwm/plot_results.py \
    --csv "${RESULTS_CSV}" \
    --outdir "${FIGS_DIR}" || { echo "[PLOT FAIL]"; FAILS=$((FAILS+1)); }

  python cpwm/analysis_contact_probe.py \
    --csv "${RESULTS_CSV}" \
    --out "${ANALYSIS_CSV}" || { echo "[ANALYSIS FAIL]"; FAILS=$((FAILS+1)); }

  python cpwm/analysis_tactile_error.py \
    --runs_dir "${RUNS_DIR}" \
    --results_csv "${RESULTS_CSV}" \
    --out_csv "${TACTILE_ERROR_CSV}" \
    --out_fig "${TACTILE_ERROR_FIG}" || { echo "[TACTILE ERROR ANALYSIS FAIL]"; FAILS=$((FAILS+1)); }

  python cpwm/analysis_dynamics_summary.py \
    --csv "${RESULTS_CSV}" \
    --out "${DYNAMICS_CSV}" || { echo "[DYNAMICS ANALYSIS FAIL]"; FAILS=$((FAILS+1)); }

  [[ -s "${RESULTS_CSV}" ]] || { echo "[MISSING RESULTS CSV] ${RESULTS_CSV}"; FAILS=$((FAILS+1)); }
  [[ -f "${ANALYSIS_CSV}" ]] || { echo "[MISSING ANALYSIS CSV] ${ANALYSIS_CSV}"; FAILS=$((FAILS+1)); }
  [[ -f "${TACTILE_ERROR_CSV}" ]] || { echo "[MISSING TACTILE ERROR CSV] ${TACTILE_ERROR_CSV}"; FAILS=$((FAILS+1)); }
  [[ -f "${TACTILE_ERROR_FIG}" ]] || { echo "[MISSING TACTILE ERROR FIG] ${TACTILE_ERROR_FIG}"; FAILS=$((FAILS+1)); }
  [[ -f "${DYNAMICS_CSV}" ]] || { echo "[MISSING DYNAMICS CSV] ${DYNAMICS_CSV}"; FAILS=$((FAILS+1)); }

  shopt -s nullglob
  fig_files=("${FIGS_DIR}"/*.png)
  shopt -u nullglob
  if [[ ${#fig_files[@]} -eq 0 ]]; then
    echo "[MISSING FIGURES] ${FIGS_DIR}"
    FAILS=$((FAILS+1))
  fi
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

if [[ "$FAILS" -ne 0 ]]; then
  exit 1
fi

exit 0
