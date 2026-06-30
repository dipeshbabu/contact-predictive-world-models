#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PPO_TASKS=${PPO_TASKS:-"h1touch-walk-v0 h1touch-push-v0 h1touch-door-v0"}
SEEDS=${SEEDS:-"0 1 2"}
PPO_SWEEP_CONFIGS=${PPO_SWEEP_CONFIGS:-"balanced large_entropy long_rollout"}
PPO_TRAIN_STEPS=${PPO_TRAIN_STEPS:-1000000}
PPO_EVAL_EPISODES=${PPO_EVAL_EPISODES:-20}
PPO_NUM_ENVS=${PPO_NUM_ENVS:-4}
RUNS_DIR=${RUNS_DIR:-outputs/runs_ppo_sweep}
RESULTS_CSV=${RESULTS_CSV:-outputs/results/ppo_sweep.csv}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$RUNS_DIR" "$(dirname "$RESULTS_CSV")"

ppo_config_args () {
  case "$1" in
    balanced)
      echo "--learning_rate 0.0003 --n_steps 2048 --batch_size 1024 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 --ent_coef 0.0 --net_width 256 --net_depth 2"
      ;;
    large_entropy)
      echo "--learning_rate 0.00025 --n_steps 2048 --batch_size 1024 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 --ent_coef 0.01 --net_width 512 --net_depth 2"
      ;;
    long_rollout)
      echo "--learning_rate 0.0002 --n_steps 4096 --batch_size 2048 --gamma 0.995 --gae_lambda 0.97 --clip_range 0.15 --ent_coef 0.003 --net_width 512 --net_depth 3"
      ;;
    *)
      echo "Unknown PPO sweep config: $1" >&2
      return 1
      ;;
  esac
}

for env in $PPO_TASKS; do
  for seed in $SEEDS; do
    for sensors_name in proprio tactile; do
      sensors=""
      [[ "$sensors_name" == "tactile" ]] && sensors="tactile"
      for cfg in $PPO_SWEEP_CONFIGS; do
        run_dir="${RUNS_DIR}/${env}_ppo_${sensors_name}_${cfg}_s${seed}"
        read -r -a cfg_args <<< "$(ppo_config_args "$cfg")"
        args=(
          --env "$env"
          --seed "$seed"
          --steps "$PPO_TRAIN_STEPS"
          --logdir "$run_dir"
          --sensors "$sensors"
          --num_envs "$PPO_NUM_ENVS"
          "${cfg_args[@]}"
        )
        [[ "$DRY_RUN" == "1" ]] && args+=(--dry_run)
        echo "[PPO SWEEP TRAIN] env=$env seed=$seed sensors=$sensors_name config=$cfg"
        python cpwm/train_ppo.py "${args[@]}"

        eval_args=(
          --run_dir "$run_dir"
          --env "$env"
          --seed "$seed"
          --variant "ppo_${sensors_name}_${cfg}"
          --episodes "$PPO_EVAL_EPISODES"
          --sensors "$sensors"
          --results_csv "$RESULTS_CSV"
        )
        [[ "$DRY_RUN" == "1" ]] && eval_args+=(--dry_run)
        echo "[PPO SWEEP EVAL] env=$env seed=$seed sensors=$sensors_name config=$cfg"
        python cpwm/eval_ppo.py "${eval_args[@]}"
      done
    done
  done
done
