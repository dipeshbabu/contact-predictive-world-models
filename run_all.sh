#!/usr/bin/env bash
set -u
set -o pipefail

FAILS=0

LOCO_TASKS=${LOCO_TASKS:-"h1touch-walk-v0 h1touch-run-v0"}
MANIP_TASKS=${MANIP_TASKS:-"h1touch-push-v0 h1touch-door-v0 h1touch-cabinet-v0 h1touch-insert_small-v0"}
DREAMER_TASKS=${DREAMER_TASKS:-"$LOCO_TASKS $MANIP_TASKS"}
PPO_TASKS=${PPO_TASKS:-"$DREAMER_TASKS"}
DREAMER_VARIANTS=${DREAMER_VARIANTS:-"base aux recon future3 noact"}
SEEDS=${SEEDS:-"0 1 2"}

TRAIN_STEPS=${TRAIN_STEPS:-2000000}
PPO_TRAIN_STEPS=${PPO_TRAIN_STEPS:-1000000}
EVAL_STEPS=${EVAL_STEPS:-20000}
PPO_EVAL_EPISODES=${PPO_EVAL_EPISODES:-20}
NUM_ENVS=${NUM_ENVS:-4}
PPO_NUM_ENVS=${PPO_NUM_ENVS:-4}
JAX_PLATFORM=${JAX_PLATFORM:-gpu}

AUX_ON=${AUX_ON:-0.1}
AUX_OFF=${AUX_OFF:-0.0}
CONTACT_AUX_ON=${CONTACT_AUX_ON:-0.1}
TACTILE_GROUP_AUX_ON=${TACTILE_GROUP_AUX_ON:-0.05}
TACTILE_MASK_PROB=${TACTILE_MASK_PROB:-0.25}
TACTILE_AUX_GROUPS=${TACTILE_AUX_GROUPS:-8}
CONTACT_IMAGINE_ON=${CONTACT_IMAGINE_ON:-0.05}
CONTACT_AUX_ENSEMBLE=${CONTACT_AUX_ENSEMBLE:-4}
CONTACT_LABEL_OVERRIDES=${CONTACT_LABEL_OVERRIDES:-}
DOOR_SUCCESS_STAND_THRESHOLD=${DOOR_SUCCESS_STAND_THRESHOLD:-0.8}
DOOR_SUCCESS_DOOR_THRESHOLD=${DOOR_SUCCESS_DOOR_THRESHOLD:-0.8}
DOOR_SUCCESS_HATCH_THRESHOLD=${DOOR_SUCCESS_HATCH_THRESHOLD:-0.8}
DOOR_SUCCESS_PASSAGE_THRESHOLD=${DOOR_SUCCESS_PASSAGE_THRESHOLD:-0.8}
INSERT_SUCCESS_STAND_THRESHOLD=${INSERT_SUCCESS_STAND_THRESHOLD:-0.8}
INSERT_SUCCESS_CUBE_THRESHOLD=${INSERT_SUCCESS_CUBE_THRESHOLD:-0.9}
INSERT_SUCCESS_PEG_HEIGHT_THRESHOLD=${INSERT_SUCCESS_PEG_HEIGHT_THRESHOLD:-0.9}

NOISES=${NOISES:-"0.0 0.01 0.02 0.05"}
DROPS=${DROPS:-"0.0 0.2 0.4 0.6"}
MASS_SCALES=${MASS_SCALES:-"1.0 0.9 1.1"}
FRICTION_SCALES=${FRICTION_SCALES:-"1.0 0.8 1.2"}
RUN_PPO=${RUN_PPO:-1}
RUN_PPO_TACTILE=${RUN_PPO_TACTILE:-1}
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
echo "DREAMER_VARIANTS=$DREAMER_VARIANTS"
echo "SEEDS=$SEEDS"
echo "TRAIN_STEPS=$TRAIN_STEPS"
echo "PPO_TRAIN_STEPS=$PPO_TRAIN_STEPS"
echo "EVAL_STEPS=$EVAL_STEPS"
echo "PPO_EVAL_EPISODES=$PPO_EVAL_EPISODES"
echo "NUM_ENVS=$NUM_ENVS"
echo "PPO_NUM_ENVS=$PPO_NUM_ENVS"
echo "JAX_PLATFORM=$JAX_PLATFORM"
echo "RUN_PPO_TACTILE=$RUN_PPO_TACTILE"
echo "DRY_RUN=$DRY_RUN"

train_dreamer_one () {
  local env="$1"
  local variant="$2"
  local seed="$3"
  local aux_weight="$4"
  local aux_mode="$5"
  local aux_horizon="$6"
  local aux_action="$7"
  local contact_weight="$8"
  local contact_mode="$9"
  local contact_horizon="${10}"
  local contact_action="${11}"
  local sensors="${12}"
  local logdir="${RUNS_DIR}/${env}_${variant}_s${seed}"
  local args=(
    --env "${env}"
    --seed "${seed}"
    --steps "${TRAIN_STEPS}"
    --num_envs "${NUM_ENVS}"
    --jax_platform "${JAX_PLATFORM}"
    --tactile_aux_weight "${aux_weight}"
    --tactile_aux_mode "${aux_mode}"
    --tactile_aux_horizon "${aux_horizon}"
    --contact_aux_weight "${contact_weight}"
    --contact_aux_mode "${contact_mode}"
    --contact_aux_horizon "${contact_horizon}"
    --contact_label_overrides "${CONTACT_LABEL_OVERRIDES}"
    --door_success_stand_threshold "${DOOR_SUCCESS_STAND_THRESHOLD}"
    --door_success_door_threshold "${DOOR_SUCCESS_DOOR_THRESHOLD}"
    --door_success_hatch_threshold "${DOOR_SUCCESS_HATCH_THRESHOLD}"
    --door_success_passage_threshold "${DOOR_SUCCESS_PASSAGE_THRESHOLD}"
    --insert_success_stand_threshold "${INSERT_SUCCESS_STAND_THRESHOLD}"
    --insert_success_cube_threshold "${INSERT_SUCCESS_CUBE_THRESHOLD}"
    --insert_success_peg_height_threshold "${INSERT_SUCCESS_PEG_HEIGHT_THRESHOLD}"
    --sensors "${sensors}"
    --logdir "${logdir}"
  )

  if [[ "${aux_action}" != "1" ]]; then
    args+=(--no_tactile_aux_action)
  fi
  if [[ "${contact_action}" != "1" ]]; then
    args+=(--no_contact_aux_action)
  fi
  case "$variant" in
    masked|frontier|frontier_rgb)
      args+=(--tactile_mask_prob "${TACTILE_MASK_PROB}")
      ;;
  esac
  case "$variant" in
    tactile_group|frontier|frontier_rgb)
      args+=(
        --tactile_group_aux_weight "${TACTILE_GROUP_AUX_ON}"
        --tactile_aux_groups "${TACTILE_AUX_GROUPS}"
      )
      ;;
  esac
  case "$variant" in
    contact_frontier|frontier|frontier_rgb)
      args+=(
        --contact_aux_balanced
        --contact_aux_rich_labels
        --contact_aux_ensemble "${CONTACT_AUX_ENSEMBLE}"
        --contact_imagine_weight "${CONTACT_IMAGINE_ON}"
      )
      ;;
  esac

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

dreamer_variant_config () {
  local variant="$1"
  case "$variant" in
    proprio|proprio_only)
      echo "${AUX_OFF} future 1 1 ${AUX_OFF} future 1 1 __none__"
      ;;
    base)
      echo "${AUX_OFF} future 1 1 ${AUX_OFF} future 1 1 tactile"
      ;;
    aux|future1)
      echo "${AUX_ON} future 1 1 ${AUX_OFF} future 1 1 tactile"
      ;;
    recon|current)
      echo "${AUX_ON} current 1 1 ${AUX_OFF} future 1 1 tactile"
      ;;
    future3)
      echo "${AUX_ON} future 3 1 ${AUX_OFF} future 1 1 tactile"
      ;;
    future5)
      echo "${AUX_ON} future 5 1 ${AUX_OFF} future 1 1 tactile"
      ;;
    masked)
      echo "${AUX_ON} masked 1 1 ${AUX_OFF} future 1 1 tactile"
      ;;
    tactile_group)
      echo "${AUX_ON} future 1 1 ${AUX_OFF} future 1 1 tactile"
      ;;
    noact|future1_noact)
      echo "${AUX_ON} future 1 0 ${AUX_OFF} future 1 1 tactile"
      ;;
    contact|contact1)
      echo "${AUX_OFF} future 1 1 ${CONTACT_AUX_ON} future 1 1 tactile"
      ;;
    contact3)
      echo "${AUX_OFF} future 1 1 ${CONTACT_AUX_ON} future 3 1 tactile"
      ;;
    contact_onset|onset)
      echo "${AUX_OFF} future 1 1 ${CONTACT_AUX_ON} onset 1 1 tactile"
      ;;
    contact_change|change)
      echo "${AUX_OFF} future 1 1 ${CONTACT_AUX_ON} change 1 1 tactile"
      ;;
    contact_frontier)
      echo "${AUX_OFF} future 1 1 ${CONTACT_AUX_ON} onset 1 1 tactile"
      ;;
    both|aux_contact)
      echo "${AUX_ON} future 1 1 ${CONTACT_AUX_ON} future 1 1 tactile"
      ;;
    both_onset|aux_contact_onset)
      echo "${AUX_ON} future 1 1 ${CONTACT_AUX_ON} onset 1 1 tactile"
      ;;
    both_change|aux_contact_change)
      echo "${AUX_ON} future 1 1 ${CONTACT_AUX_ON} change 1 1 tactile"
      ;;
    frontier)
      echo "${AUX_ON} masked 1 1 ${CONTACT_AUX_ON} onset 1 1 tactile"
      ;;
    frontier_rgb)
      echo "${AUX_ON} masked 1 1 ${CONTACT_AUX_ON} onset 1 1 tactile,image"
      ;;
    *)
      echo "Unknown DREAMER variant: ${variant}" >&2
      return 1
      ;;
  esac
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
    --contact_label_overrides "${CONTACT_LABEL_OVERRIDES}"
    --door_success_stand_threshold "${DOOR_SUCCESS_STAND_THRESHOLD}"
    --door_success_door_threshold "${DOOR_SUCCESS_DOOR_THRESHOLD}"
    --door_success_hatch_threshold "${DOOR_SUCCESS_HATCH_THRESHOLD}"
    --door_success_passage_threshold "${DOOR_SUCCESS_PASSAGE_THRESHOLD}"
    --insert_success_stand_threshold "${INSERT_SUCCESS_STAND_THRESHOLD}"
    --insert_success_cube_threshold "${INSERT_SUCCESS_CUBE_THRESHOLD}"
    --insert_success_peg_height_threshold "${INSERT_SUCCESS_PEG_HEIGHT_THRESHOLD}"
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
  local variant="$3"
  local sensors="$4"
  local logdir="${RUNS_DIR}/${env}_${variant}_s${seed}"
  local args=(
    --env "${env}"
    --seed "${seed}"
    --steps "${PPO_TRAIN_STEPS}"
    --logdir "${logdir}"
    --sensors "${sensors}"
    --num_envs "${PPO_NUM_ENVS}"
    --contact_label_overrides "${CONTACT_LABEL_OVERRIDES}"
    --door_success_stand_threshold "${DOOR_SUCCESS_STAND_THRESHOLD}"
    --door_success_door_threshold "${DOOR_SUCCESS_DOOR_THRESHOLD}"
    --door_success_hatch_threshold "${DOOR_SUCCESS_HATCH_THRESHOLD}"
    --door_success_passage_threshold "${DOOR_SUCCESS_PASSAGE_THRESHOLD}"
    --insert_success_stand_threshold "${INSERT_SUCCESS_STAND_THRESHOLD}"
    --insert_success_cube_threshold "${INSERT_SUCCESS_CUBE_THRESHOLD}"
    --insert_success_peg_height_threshold "${INSERT_SUCCESS_PEG_HEIGHT_THRESHOLD}"
  )

  if [[ "$DRY_RUN" != "1" && ( -f "${logdir}/ppo_model.zip" || -f "${logdir}/ppo_model" ) ]]; then
    echo "[SKIP PPO TRAIN] ${logdir}"
    return 0
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry_run)
  fi

  echo "[PPO TRAIN] env=${env} variant=${variant} seed=${seed}"
  python cpwm/train_ppo.py "${args[@]}"
}

eval_ppo_one () {
  local run_dir="$1"
  local env="$2"
  local seed="$3"
  local variant="$4"
  local sensors="$5"
  local noise="$6"
  local drop="$7"
  local mass="$8"
  local fric="$9"
  local args=(
    --run_dir "${run_dir}"
    --env "${env}"
    --seed "${seed}"
    --variant "${variant}"
    --episodes "${PPO_EVAL_EPISODES}"
    --sensors "${sensors}"
    --noise "${noise}"
    --tactile_dropout "${drop}"
    --mass_scale "${mass}"
    --friction_scale "${fric}"
    --results_csv "${RESULTS_CSV}"
    --contact_label_overrides "${CONTACT_LABEL_OVERRIDES}"
    --door_success_stand_threshold "${DOOR_SUCCESS_STAND_THRESHOLD}"
    --door_success_door_threshold "${DOOR_SUCCESS_DOOR_THRESHOLD}"
    --door_success_hatch_threshold "${DOOR_SUCCESS_HATCH_THRESHOLD}"
    --door_success_passage_threshold "${DOOR_SUCCESS_PASSAGE_THRESHOLD}"
    --insert_success_stand_threshold "${INSERT_SUCCESS_STAND_THRESHOLD}"
    --insert_success_cube_threshold "${INSERT_SUCCESS_CUBE_THRESHOLD}"
    --insert_success_peg_height_threshold "${INSERT_SUCCESS_PEG_HEIGHT_THRESHOLD}"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    mkdir -p "${run_dir}"
    args+=(--dry_run)
  fi

  echo "[PPO EVAL] ${run_dir} variant=${variant} noise=${noise} drop=${drop} mass=${mass} fric=${fric}"
  python cpwm/eval_ppo.py "${args[@]}"
}

for env in $DREAMER_TASKS; do
  for seed in $SEEDS; do
    for variant in $DREAMER_VARIANTS; do
      read -r aux_weight aux_mode aux_horizon aux_action contact_weight contact_mode contact_horizon contact_action sensors < <(dreamer_variant_config "$variant") || { FAILS=$((FAILS+1)); continue; }
      [[ "$sensors" == "__none__" ]] && sensors=""
      train_dreamer_one "$env" "$variant" "$seed" "$aux_weight" "$aux_mode" "$aux_horizon" "$aux_action" "$contact_weight" "$contact_mode" "$contact_horizon" "$contact_action" "$sensors" || { echo "[DREAMER TRAIN FAIL] $env $variant s$seed"; FAILS=$((FAILS+1)); }
    done
  done
done

for env in $DREAMER_TASKS; do
  for seed in $SEEDS; do
    for variant in $DREAMER_VARIANTS; do
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
      fi
    done
  done
done

if [[ "$RUN_PPO" == "1" ]]; then
  PPO_VARIANTS="ppo_proprio"
  if [[ "$RUN_PPO_TACTILE" == "1" ]]; then
    PPO_VARIANTS="$PPO_VARIANTS ppo_tactile"
  fi
  for env in $PPO_TASKS; do
    for seed in $SEEDS; do
      for variant in $PPO_VARIANTS; do
        sensors=""
        [[ "$variant" == "ppo_tactile" ]] && sensors="tactile"
        train_ppo_one "$env" "$seed" "$variant" "$sensors" || { echo "[PPO TRAIN FAIL] $env $variant s$seed"; FAILS=$((FAILS+1)); }
      done
    done
  done

  for env in $PPO_TASKS; do
    for seed in $SEEDS; do
      for variant in $PPO_VARIANTS; do
        sensors=""
        [[ "$variant" == "ppo_tactile" ]] && sensors="tactile"
        run_dir="${RUNS_DIR}/${env}_${variant}_s${seed}"
        if [[ "$DRY_RUN" != "1" && ! -d "${run_dir}" ]]; then
          echo "[SKIP PPO EVAL] missing run dir ${run_dir}"
          FAILS=$((FAILS+1))
          continue
        fi

        for noise in $NOISES; do
          for drop in $DROPS; do
            eval_ppo_one "$run_dir" "$env" "$seed" "$variant" "$sensors" "$noise" "$drop" "1.0" "1.0" || { echo "[PPO SENSORY FAIL] $run_dir noise=$noise drop=$drop"; FAILS=$((FAILS+1)); }
          done
        done

        if [[ "$RUN_DYNAMICS" == "1" ]]; then
          for mass in $MASS_SCALES; do
            for fric in $FRICTION_SCALES; do
              [[ "$mass" == "1.0" && "$fric" == "1.0" ]] && continue
              eval_ppo_one "$run_dir" "$env" "$seed" "$variant" "$sensors" "0.0" "0.0" "$mass" "$fric" || { echo "[PPO DYNAMICS FAIL] $run_dir mass=$mass fric=$fric"; FAILS=$((FAILS+1)); }
            done
          done
        fi
      done
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
