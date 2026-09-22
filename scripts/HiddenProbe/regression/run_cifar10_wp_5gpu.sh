#!/usr/bin/env bash
set -euo pipefail

# CIFAR10-WP HiddenProbe: submit 5 independent Slurm jobs, one seed / one GPU.
#
# Usage:
#   bash scripts/HiddenProbe/regression/run_cifar10_wp_5gpu.sh
#   bash scripts/HiddenProbe/regression/run_cifar10_wp_5gpu.sh 128
#
# Each seed requests ONE GPU, so the five seeds do not need to land on the same node.
#
# Optional Slurm overrides:
#   SLURM_TIME=2-00:00:00
#   SLURM_CPUS=4
#   SLURM_MEM=24G
#   SLURM_PARTITION=<partition>
#   SLURM_ACCOUNT=<account>
#
# Internal worker mode:
#   SEED_ONLY=0 bash scripts/HiddenProbe/regression/run_cifar10_wp_5gpu.sh 128

N_PROBES="${1:-128}"
NUM_SEEDS=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
OUT_ROOT="${OUT_DIR:-checkpoints_rerun}"

WP_DIR="${WP_DIR:-$DATA_ROOT/regression/cifar10_wp}"
CNN_CACHE="${CNN_CACHE:-$WP_DIR/wp_cnn_cache}"
SPLITS="$WP_DIR/splits.json"

SLURM_TIME="${SLURM_TIME:-2-00:00:00}"
SLURM_CPUS="${SLURM_CPUS:-4}"
SLURM_MEM="${SLURM_MEM:-24G}"

mkdir -p "$OUT_ROOT/logs"

# ---------------------------------------------------------------------------
# Worker mode: one seed on one allocated GPU.
# ---------------------------------------------------------------------------
if [[ -n "${SEED_ONLY:-}" ]]; then
  SEED="$SEED_ONLY"

  if [[ "$SEED" -lt 0 || "$SEED" -ge "$NUM_SEEDS" ]]; then
    echo "Invalid SEED_ONLY=$SEED"
    exit 2
  fi

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

  EXP_NAME="hiddenprobe_cifar10_wp_Q${N_PROBES}_s${SEED}"
  SEED_OUT_DIR="$OUT_ROOT/$EXP_NAME"

  echo "============================================================"
  echo "CIFAR10-WP HiddenProbe"
  echo "seed:    $SEED"
  echo "job:     ${SLURM_JOB_ID:-none}"
  echo "gpu(s):  ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "output:  $SEED_OUT_DIR"
  echo "============================================================"

  python main.py     --method hiddenprobe     --task regression     --dataset cifar10_wp     --zoo wp     --cnn_cache "$CNN_CACHE"     --splits "$SPLITS"     --hidden_mode on     --probe_sharing shared     --assert_canonical 1     --target_space raw     --gen_type deep_linear_5     --adapter_preset compact     --interaction_rank 32     --mixer_hidden 256     --hidden_dim 0     --n_probes "$N_PROBES"     --lr "$LR"     --probe_lr "$PROBE_LR"     --batch_size 32     --rank_loss_w 0.0     --warmup 0     --grad_clip 0     --scheduler plateau     --plateau_factor 0.5     --plateau_patience 4     --plateau_min_lr 3e-5     --probe_min_lr 3e-6     --epochs 18     --sched_total_epochs 30     --n_train 0     --eval_every 1500     --eval_cnn_bs 256     --seed "$SEED"     --exp_name "$EXP_NAME"     --out_dir "$SEED_OUT_DIR"     --dump_preds "$SEED_OUT_DIR/preds.npz"

  exit 0
fi

# ---------------------------------------------------------------------------
# Submission mode: submit five independent 1-GPU jobs.
# ---------------------------------------------------------------------------
command -v sbatch >/dev/null 2>&1 || {
  echo "ERROR: sbatch was not found."
  exit 2
}

[[ -s "$SPLITS" ]] || {
  echo "Missing split file: $SPLITS"
  echo "Run first:"
  echo "  bash scripts/setup_data/regression_cifar10_wp.sh"
  exit 2
}

for s in train val test; do
  [[ -s "$CNN_CACHE/cnn_cache_${s}.pt" ]] || {
    echo "Missing cache: $CNN_CACHE/cnn_cache_${s}.pt"
    echo "Run first:"
    echo "  bash scripts/setup_data/regression_cifar10_wp.sh"
    exit 2
  }
done

SBATCH_COMMON=(
  --nodes=1
  --ntasks=1
  --gres=gpu:1
  --cpus-per-task="$SLURM_CPUS"
  --mem="$SLURM_MEM"
  --time="$SLURM_TIME"
)

if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SBATCH_COMMON+=(--partition="$SLURM_PARTITION")
fi

if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  SBATCH_COMMON+=(--account="$SLURM_ACCOUNT")
fi

JOB_IDS=()

for SEED in 0 1 2 3 4; do
  EXP_NAME="hiddenprobe_cifar10_wp_Q${N_PROBES}_s${SEED}"
  LOG_FILE="$OUT_ROOT/logs/${EXP_NAME}_%j.log"

  JOB_ID="$(
    sbatch --parsable       "${SBATCH_COMMON[@]}"       --job-name="wp_s${SEED}"       --output="$LOG_FILE"       --export="ALL,SEED_ONLY=${SEED},OUT_DIR=${OUT_ROOT},DATA_ROOT=${DATA_ROOT},WP_DIR=${WP_DIR},CNN_CACHE=${CNN_CACHE}"       --wrap="cd '$REPO_ROOT' && bash '$path' '$N_PROBES'"
  )"

  JOB_IDS+=("$JOB_ID")
  echo "submitted seed $SEED -> job $JOB_ID"
done

DEPENDENCY="$(IFS=:; echo "${JOB_IDS[*]}")"

AGG_LOG="$OUT_ROOT/logs/cifar10_wp_Q${N_PROBES}_aggregate_%j.log"

AGG_JOB="$(
  sbatch --parsable     --dependency="afterok:$DEPENDENCY"     --nodes=1     --ntasks=1     --cpus-per-task=1     --mem=2G     --time=00:10:00     --job-name="wp_aggregate"     --output="$AGG_LOG"     --wrap="cd '$REPO_ROOT' && python scripts/HiddenProbe/regression/aggregate_results.py       --dataset 'CIFAR-10 Wild Park'       --model 'HiddenProbe'       '$OUT_ROOT/hiddenprobe_cifar10_wp_Q${N_PROBES}_s0/summary.json'       '$OUT_ROOT/hiddenprobe_cifar10_wp_Q${N_PROBES}_s1/summary.json'       '$OUT_ROOT/hiddenprobe_cifar10_wp_Q${N_PROBES}_s2/summary.json'       '$OUT_ROOT/hiddenprobe_cifar10_wp_Q${N_PROBES}_s3/summary.json'       '$OUT_ROOT/hiddenprobe_cifar10_wp_Q${N_PROBES}_s4/summary.json'"
)"

echo
echo "Five independent GPU jobs submitted:"
for i in "${!JOB_IDS[@]}"; do
  echo "  seed $i -> job ${JOB_IDS[$i]}"
done
echo
echo "aggregate job -> $AGG_JOB"
echo "It will start only after all five seed jobs finish successfully."
echo
echo "Monitor:"
echo "  squeue -j $(IFS=,; echo "${JOB_IDS[*]}"),$AGG_JOB"
