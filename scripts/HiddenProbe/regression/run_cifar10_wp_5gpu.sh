#!/usr/bin/env bash
set -euo pipefail

# CIFAR10-WP HiddenProbe: 5 seeds in parallel on 5 GPUs.
#
# Usage:
#   bash scripts/HiddenProbe/regression/run_cifar10_wp_5gpu.sh
#   bash scripts/HiddenProbe/regression/run_cifar10_wp_5gpu.sh 128
#
# If run outside a Slurm allocation, this script requests 5 GPUs automatically
# and re-launches itself inside the allocation.
#
# Optional Slurm overrides:
#   SLURM_TIME=2-00:00:00
#   SLURM_CPUS=20
#   SLURM_MEM=96G
#   SLURM_PARTITION=<partition>
#   SLURM_ACCOUNT=<account>
#
# Example:
#   SLURM_PARTITION=gpu bash scripts/HiddenProbe/regression/run_cifar10_wp_5gpu.sh 128

N_PROBES="${1:-128}"
NUM_SEEDS=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Allocate 5 GPUs automatically when not already inside a Slurm job.
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  command -v srun >/dev/null 2>&1 || {
    echo "ERROR: not inside a Slurm job and 'srun' was not found."
    echo "Run this script on a Slurm cluster, or allocate 5 GPUs manually first."
    exit 2
  }

  SLURM_TIME="${SLURM_TIME:-2-00:00:00}"
  SLURM_CPUS="${SLURM_CPUS:-20}"
  SLURM_MEM="${SLURM_MEM:-96G}"

  SRUN_ARGS=(
    --ntasks=1
    --nodes=1
    --gres=gpu:5
    --cpus-per-task="$SLURM_CPUS"
    --mem="$SLURM_MEM"
    --time="$SLURM_TIME"
  )

  if [[ -n "${SLURM_PARTITION:-}" ]]; then
    SRUN_ARGS+=(--partition="$SLURM_PARTITION")
  fi

  if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
    SRUN_ARGS+=(--account="$SLURM_ACCOUNT")
  fi

  echo "Requesting one Slurm node with 5 GPUs..."
  echo "  time:  $SLURM_TIME"
  echo "  cpus:  $SLURM_CPUS"
  echo "  mem:   $SLURM_MEM"
  [[ -n "${SLURM_PARTITION:-}" ]] && echo "  part:  $SLURM_PARTITION"
  [[ -n "${SLURM_ACCOUNT:-}" ]] && echo "  acct:  $SLURM_ACCOUNT"
  echo

  exec srun "${SRUN_ARGS[@]}"     bash "$0" "$@"
fi

echo "Running inside Slurm job: $SLURM_JOB_ID"

# Slurm normally remaps allocated GPUs to local CUDA ids 0..4.
# Require at least five visible devices before starting the seeds.
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$GPU_COUNT" -lt 5 ]]; then
    echo "ERROR: Slurm job has only $GPU_COUNT visible GPU(s); 5 are required."
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    exit 2
  fi
else
  echo "WARNING: nvidia-smi not found; cannot verify that 5 GPUs are visible."
fi

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
OUT_ROOT="${OUT_DIR:-checkpoints_rerun}"

WP_DIR="${WP_DIR:-$DATA_ROOT/regression/cifar10_wp}"
CNN_CACHE="${CNN_CACHE:-$WP_DIR/wp_cnn_cache}"
SPLITS="$WP_DIR/splits.json"

# Setup once BEFORE launching the five jobs.
bash scripts/setup_data/regression_cifar10_wp.sh

[[ -s "$SPLITS" ]] || {
  echo "Missing split file: $SPLITS"
  exit 2
}

for s in train val test; do
  [[ -s "$CNN_CACHE/cnn_cache_${s}.pt" ]] || {
    echo "Missing cache: $CNN_CACHE/cnn_cache_${s}.pt"
    exit 2
  }
done

if [[ "$N_PROBES" == "64" ]]; then
  LR=2e-4
  PROBE_LR=2e-4
else
  LR=3e-4
  PROBE_LR=1e-4
fi

mkdir -p "$OUT_ROOT/logs"

PIDS=()

for SEED in 0 1 2 3 4; do
  GPU=$SEED

  EXP_NAME="hiddenprobe_cifar10_wp_Q${N_PROBES}_s${SEED}"
  SEED_OUT_DIR="$OUT_ROOT/$EXP_NAME"
  LOG_FILE="$OUT_ROOT/logs/${EXP_NAME}.log"

  echo "Launching seed=$SEED on local GPU=$GPU"
  echo "  output: $SEED_OUT_DIR"
  echo "  log:    $LOG_FILE"

  CUDA_VISIBLE_DEVICES="$GPU"   python main.py     --method hiddenprobe     --task regression     --dataset cifar10_wp     --zoo wp     --cnn_cache "$CNN_CACHE"     --splits "$SPLITS"     --hidden_mode on     --probe_sharing shared     --assert_canonical 1     --target_space raw     --gen_type deep_linear_5     --adapter_preset compact     --interaction_rank 32     --mixer_hidden 256     --hidden_dim 0     --n_probes "$N_PROBES"     --lr "$LR"     --probe_lr "$PROBE_LR"     --batch_size 32     --rank_loss_w 0.0     --warmup 0     --grad_clip 0     --scheduler plateau     --plateau_factor 0.5     --plateau_patience 4     --plateau_min_lr 3e-5     --probe_min_lr 3e-6     --epochs 18     --sched_total_epochs 30     --n_train 0     --eval_every 1500     --eval_cnn_bs 256     --seed "$SEED"     --exp_name "$EXP_NAME"     --out_dir "$SEED_OUT_DIR"     --dump_preds "$SEED_OUT_DIR/preds.npz"     > "$LOG_FILE" 2>&1 &

  PIDS+=($!)
done

echo
echo "All 5 jobs launched:"
for SEED in 0 1 2 3 4; do
  echo "  seed $SEED -> local GPU $SEED"
done
echo

FAILED=0

for i in "${!PIDS[@]}"; do
  SEED=$i
  if wait "${PIDS[$i]}"; then
    echo "seed $SEED finished successfully"
  else
    echo "seed $SEED FAILED"
    FAILED=1
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "At least one seed failed. Check $OUT_ROOT/logs/"
  exit 1
fi

python scripts/HiddenProbe/regression/aggregate_results.py   --dataset "CIFAR-10 Wild Park"   --model "HiddenProbe"   "$OUT_ROOT/hiddenprobe_cifar10_wp_Q${N_PROBES}_s0/summary.json"   "$OUT_ROOT/hiddenprobe_cifar10_wp_Q${N_PROBES}_s1/summary.json"   "$OUT_ROOT/hiddenprobe_cifar10_wp_Q${N_PROBES}_s2/summary.json"   "$OUT_ROOT/hiddenprobe_cifar10_wp_Q${N_PROBES}_s3/summary.json"   "$OUT_ROOT/hiddenprobe_cifar10_wp_Q${N_PROBES}_s4/summary.json"
