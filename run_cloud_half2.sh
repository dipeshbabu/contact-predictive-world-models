#!/usr/bin/env bash
# ============================================================
# Half2 Cloud GPU Script - Exactly like Mason
# Tasks: run, door, insert_small
# Seeds: 0, 1, 2 (same as Mason)
# Dreamer: 2,000,000 steps | PPO: 1,000,000 steps | Eval: 20,000 steps
# Requires: 8x GPU machine
# ============================================================
set -u

source .venv/bin/activate
mkdir -p logs outputs/runs_half2 outputs/results outputs/figs_half2

RUNS_DIR="outputs/runs_half2"
RESULTS_CSV="outputs/results/results_half2.csv"

echo "======================================================"
echo " Starting Half2 Cloud Training - All 27 runs"
echo " $(date)"
echo "======================================================"

# Check GPUs
nvidia-smi --query-gpu=name --format=csv,noheader

# ============================================================
# BATCH 1: 8 dreamer jobs (run base/aux s0/s1/s2, door base)
# ============================================================
echo ""
echo "=== BATCH 1: 8 Dreamer jobs starting... ==="
echo "$(date)"

CUDA_VISIBLE_DEVICES=0 python cpwm/train_dreamer.py \
  --env h1touch-run-v0 --seed 0 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.0 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-run-v0_base_s0 > logs/run_base_s0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python cpwm/train_dreamer.py \
  --env h1touch-run-v0 --seed 1 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.0 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-run-v0_base_s1 > logs/run_base_s1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python cpwm/train_dreamer.py \
  --env h1touch-run-v0 --seed 2 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.0 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-run-v0_base_s2 > logs/run_base_s2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python cpwm/train_dreamer.py \
  --env h1touch-run-v0 --seed 0 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.1 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-run-v0_aux_s0 > logs/run_aux_s0.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python cpwm/train_dreamer.py \
  --env h1touch-run-v0 --seed 1 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.1 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-run-v0_aux_s1 > logs/run_aux_s1.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python cpwm/train_dreamer.py \
  --env h1touch-run-v0 --seed 2 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.1 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-run-v0_aux_s2 > logs/run_aux_s2.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python cpwm/train_dreamer.py \
  --env h1touch-door-v0 --seed 0 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.0 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-door-v0_base_s0 > logs/door_base_s0.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python cpwm/train_dreamer.py \
  --env h1touch-door-v0 --seed 1 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.0 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-door-v0_base_s1 > logs/door_base_s1.log 2>&1 &

wait
echo "=== BATCH 1 DONE: $(date) ==="

# ============================================================
# BATCH 2: 8 dreamer jobs (door base/aux, insert base)
# ============================================================
echo ""
echo "=== BATCH 2: 8 Dreamer jobs starting... ==="
echo "$(date)"

CUDA_VISIBLE_DEVICES=0 python cpwm/train_dreamer.py \
  --env h1touch-door-v0 --seed 2 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.0 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-door-v0_base_s2 > logs/door_base_s2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python cpwm/train_dreamer.py \
  --env h1touch-door-v0 --seed 0 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.1 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-door-v0_aux_s0 > logs/door_aux_s0.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python cpwm/train_dreamer.py \
  --env h1touch-door-v0 --seed 1 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.1 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-door-v0_aux_s1 > logs/door_aux_s1.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python cpwm/train_dreamer.py \
  --env h1touch-door-v0 --seed 2 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.1 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-door-v0_aux_s2 > logs/door_aux_s2.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python cpwm/train_dreamer.py \
  --env h1touch-insert_small-v0 --seed 0 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.0 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-insert_small-v0_base_s0 > logs/insert_base_s0.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python cpwm/train_dreamer.py \
  --env h1touch-insert_small-v0 --seed 1 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.0 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-insert_small-v0_base_s1 > logs/insert_base_s1.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python cpwm/train_dreamer.py \
  --env h1touch-insert_small-v0 --seed 2 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.0 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-insert_small-v0_base_s2 > logs/insert_base_s2.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python cpwm/train_dreamer.py \
  --env h1touch-insert_small-v0 --seed 0 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.1 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-insert_small-v0_aux_s0 > logs/insert_aux_s0.log 2>&1 &

wait
echo "=== BATCH 2 DONE: $(date) ==="

# ============================================================
# BATCH 3: insert aux s1/s2 + all 9 PPO jobs
# ============================================================
echo ""
echo "=== BATCH 3: insert aux + all PPO starting... ==="
echo "$(date)"

CUDA_VISIBLE_DEVICES=0 python cpwm/train_dreamer.py \
  --env h1touch-insert_small-v0 --seed 1 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.1 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-insert_small-v0_aux_s1 > logs/insert_aux_s1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python cpwm/train_dreamer.py \
  --env h1touch-insert_small-v0 --seed 2 --steps 2000000 --num_envs 4 \
  --tactile_aux_weight 0.1 --jax_platform gpu \
  --logdir $RUNS_DIR/h1touch-insert_small-v0_aux_s2 > logs/insert_aux_s2.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python cpwm/train_ppo.py \
  --env h1touch-run-v0 --seed 0 --steps 1000000 --num_envs 4 \
  --logdir $RUNS_DIR/h1touch-run-v0_ppo_proprio_s0 > logs/run_ppo_s0.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python cpwm/train_ppo.py \
  --env h1touch-run-v0 --seed 1 --steps 1000000 --num_envs 4 \
  --logdir $RUNS_DIR/h1touch-run-v0_ppo_proprio_s1 > logs/run_ppo_s1.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python cpwm/train_ppo.py \
  --env h1touch-run-v0 --seed 2 --steps 1000000 --num_envs 4 \
  --logdir $RUNS_DIR/h1touch-run-v0_ppo_proprio_s2 > logs/run_ppo_s2.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python cpwm/train_ppo.py \
  --env h1touch-door-v0 --seed 0 --steps 1000000 --num_envs 4 \
  --logdir $RUNS_DIR/h1touch-door-v0_ppo_proprio_s0 > logs/door_ppo_s0.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python cpwm/train_ppo.py \
  --env h1touch-door-v0 --seed 1 --steps 1000000 --num_envs 4 \
  --logdir $RUNS_DIR/h1touch-door-v0_ppo_proprio_s1 > logs/door_ppo_s1.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python cpwm/train_ppo.py \
  --env h1touch-door-v0 --seed 2 --steps 1000000 --num_envs 4 \
  --logdir $RUNS_DIR/h1touch-door-v0_ppo_proprio_s2 > logs/door_ppo_s2.log 2>&1 &

wait
echo "=== BATCH 3 DONE: $(date) ==="

# ============================================================
# BATCH 4: remaining insert PPO jobs
# ============================================================
echo ""
echo "=== BATCH 4: Insert PPO jobs ==="
echo "$(date)"

CUDA_VISIBLE_DEVICES=0 python cpwm/train_ppo.py \
  --env h1touch-insert_small-v0 --seed 0 --steps 1000000 --num_envs 4 \
  --logdir $RUNS_DIR/h1touch-insert_small-v0_ppo_proprio_s0 > logs/insert_ppo_s0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python cpwm/train_ppo.py \
  --env h1touch-insert_small-v0 --seed 1 --steps 1000000 --num_envs 4 \
  --logdir $RUNS_DIR/h1touch-insert_small-v0_ppo_proprio_s1 > logs/insert_ppo_s1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python cpwm/train_ppo.py \
  --env h1touch-insert_small-v0 --seed 2 --steps 1000000 --num_envs 4 \
  --logdir $RUNS_DIR/h1touch-insert_small-v0_ppo_proprio_s2 > logs/insert_ppo_s2.log 2>&1 &

wait
echo "=== BATCH 4 DONE: $(date) ==="
echo ""
echo "======================================================"
echo " ALL TRAINING COMPLETE!"
echo " $(date)"
echo "======================================================"

# ============================================================
# EVAL PHASE: Run all eval conditions in parallel on 8 GPUs
# ============================================================
echo ""
echo "=== STARTING EVAL PHASE ==="

NOISES="0.0 0.01 0.02 0.05"
DROPS="0.0 0.2 0.4 0.6"
MASS_SCALES="0.9 1.0 1.1"
FRICTION_SCALES="0.8 1.0 1.2"

# Build list of all dreamer run dirs
DREAMER_RUNS=(
  "$RUNS_DIR/h1touch-run-v0_base_s0"
  "$RUNS_DIR/h1touch-run-v0_base_s1"
  "$RUNS_DIR/h1touch-run-v0_base_s2"
  "$RUNS_DIR/h1touch-run-v0_aux_s0"
  "$RUNS_DIR/h1touch-run-v0_aux_s1"
  "$RUNS_DIR/h1touch-run-v0_aux_s2"
  "$RUNS_DIR/h1touch-door-v0_base_s0"
  "$RUNS_DIR/h1touch-door-v0_base_s1"
  "$RUNS_DIR/h1touch-door-v0_base_s2"
  "$RUNS_DIR/h1touch-door-v0_aux_s0"
  "$RUNS_DIR/h1touch-door-v0_aux_s1"
  "$RUNS_DIR/h1touch-door-v0_aux_s2"
  "$RUNS_DIR/h1touch-insert_small-v0_base_s0"
  "$RUNS_DIR/h1touch-insert_small-v0_base_s1"
  "$RUNS_DIR/h1touch-insert_small-v0_base_s2"
  "$RUNS_DIR/h1touch-insert_small-v0_aux_s0"
  "$RUNS_DIR/h1touch-insert_small-v0_aux_s1"
  "$RUNS_DIR/h1touch-insert_small-v0_aux_s2"
)

# Generate all eval jobs and run 8 at a time
GPU_ID=0
JOBS=()

for run_dir in "${DREAMER_RUNS[@]}"; do
  for noise in $NOISES; do
    for drop in $DROPS; do
      # noise+dropout sweep (mass=1.0, fric=1.0)
      CUDA_VISIBLE_DEVICES=$GPU_ID python cpwm/eval_dreamer.py \
        --run_dir "$run_dir" --steps 20000 \
        --noise $noise --tactile_dropout $drop \
        --mass_scale 1.0 --friction_scale 1.0 \
        --results_csv "$RESULTS_CSV" >> logs/eval_dreamer.log 2>&1 &
      JOBS+=($!)
      GPU_ID=$(( (GPU_ID + 1) % 8 ))
      if [ ${#JOBS[@]} -ge 8 ]; then
        wait "${JOBS[@]}"
        JOBS=()
      fi
    done
  done
  for mass in $MASS_SCALES; do
    for fric in $FRICTION_SCALES; do
      # mass+friction sweep (noise=0.0, drop=0.0)
      CUDA_VISIBLE_DEVICES=$GPU_ID python cpwm/eval_dreamer.py \
        --run_dir "$run_dir" --steps 20000 \
        --noise 0.0 --tactile_dropout 0.0 \
        --mass_scale $mass --friction_scale $fric \
        --results_csv "$RESULTS_CSV" >> logs/eval_dreamer.log 2>&1 &
      JOBS+=($!)
      GPU_ID=$(( (GPU_ID + 1) % 8 ))
      if [ ${#JOBS[@]} -ge 8 ]; then
        wait "${JOBS[@]}"
        JOBS=()
      fi
    done
  done
done
wait
echo "=== DREAMER EVAL DONE: $(date) ==="

# PPO eval
echo "=== PPO EVAL starting ==="
PPO_RUNS=(
  "h1touch-run-v0 $RUNS_DIR/h1touch-run-v0_ppo_proprio_s0 0"
  "h1touch-run-v0 $RUNS_DIR/h1touch-run-v0_ppo_proprio_s1 1"
  "h1touch-run-v0 $RUNS_DIR/h1touch-run-v0_ppo_proprio_s2 2"
  "h1touch-door-v0 $RUNS_DIR/h1touch-door-v0_ppo_proprio_s0 0"
  "h1touch-door-v0 $RUNS_DIR/h1touch-door-v0_ppo_proprio_s1 1"
  "h1touch-door-v0 $RUNS_DIR/h1touch-door-v0_ppo_proprio_s2 2"
  "h1touch-insert_small-v0 $RUNS_DIR/h1touch-insert_small-v0_ppo_proprio_s0 0"
  "h1touch-insert_small-v0 $RUNS_DIR/h1touch-insert_small-v0_ppo_proprio_s1 1"
  "h1touch-insert_small-v0 $RUNS_DIR/h1touch-insert_small-v0_ppo_proprio_s2 2"
)

GPU_ID=0
JOBS=()
for entry in "${PPO_RUNS[@]}"; do
  env=$(echo $entry | awk '{print $1}')
  run_dir=$(echo $entry | awk '{print $2}')
  seed=$(echo $entry | awk '{print $3}')
  # noise+dropout sweep (mass=1.0, fric=1.0)
  for noise in $NOISES; do
    for drop in $DROPS; do
      CUDA_VISIBLE_DEVICES=$GPU_ID python cpwm/eval_ppo.py \
        --env $env --run_dir "$run_dir" --seed $seed --episodes 20 \
        --noise $noise --tactile_dropout $drop \
        --mass_scale 1.0 --friction_scale 1.0 \
        --results_csv "$RESULTS_CSV" >> logs/eval_ppo.log 2>&1 &
      JOBS+=($!)
      GPU_ID=$(( (GPU_ID + 1) % 8 ))
      if [ ${#JOBS[@]} -ge 8 ]; then
        wait "${JOBS[@]}"
        JOBS=()
      fi
    done
  done
  # mass+friction sweep (noise=0.0, drop=0.0)
  for mass in $MASS_SCALES; do
    for fric in $FRICTION_SCALES; do
      CUDA_VISIBLE_DEVICES=$GPU_ID python cpwm/eval_ppo.py \
        --env $env --run_dir "$run_dir" --seed $seed --episodes 20 \
        --noise 0.0 --tactile_dropout 0.0 \
        --mass_scale $mass --friction_scale $fric \
        --results_csv "$RESULTS_CSV" >> logs/eval_ppo.log 2>&1 &
      JOBS+=($!)
      GPU_ID=$(( (GPU_ID + 1) % 8 ))
      if [ ${#JOBS[@]} -ge 8 ]; then
        wait "${JOBS[@]}"
        JOBS=()
      fi
    done
  done
done
wait
echo "=== PPO EVAL DONE: $(date) ==="

# ============================================================
# FINAL: Generate results CSV and analysis
# ============================================================
echo ""
echo "=== RUNNING FINAL ANALYSIS ==="
python - <<'PY'
import pandas as pd, glob, os
csvs = glob.glob("outputs/results/results_half2.csv")
if csvs:
    df = pd.read_csv(csvs[0])
    df.drop_duplicates(inplace=True)
    df.to_csv("outputs/results/results_half2.csv", index=False)
    print(f"Final CSV: {len(df)} rows saved to outputs/results/results_half2.csv")
else:
    print("No CSV found!")
PY

echo ""
echo "======================================================"
echo " ALL DONE! Exactly like Mason."
echo " Results: outputs/results/results_half2.csv"
echo " $(date)"
echo "======================================================"
