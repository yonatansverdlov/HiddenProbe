#!/usr/bin/env bash
set -euo pipefail

# CIFAR10-WP HiddenProbe probe-count sweep: Q=8,16,32,64,128,256; seed=0.
# Runs SEQUENTIALLY on the GPU already allocated to the current shell/job.
# No sbatch, srun, or GPU allocation here.
#
# Usage:
#   bash scripts/HiddenProbe/regression/run_cifar10_wp_probe_sweep_s0.sh
#
# Optional overrides:
#   OUT_DIR=/path/to/new/sweep/results
#   DATA_ROOT=/path/to/data
#   EVAL_CNN_BS=128
#
# Re-running:
#   - A Q with a valid summary.json is skipped.
#   - A Q with training_state.pt but no summary resumes from the last completed epoch.
#   - A Q without either starts from scratch.
# The CSV is rebuilt after each completed Q and on every script invocation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PROBES=(8 16 32 64 128 256)
SEED=0

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
OUT_ROOT="${OUT_DIR:-$REPO_ROOT/checkpoints/wp_probe_sweep_s0}"
WP_DIR="${WP_DIR:-$DATA_ROOT/regression/cifar10_wp}"
CNN_CACHE="${CNN_CACHE:-$WP_DIR/wp_cnn_cache}"
SPLITS="$WP_DIR/splits.json"
EVAL_CNN_BS="${EVAL_CNN_BS:-256}"
CSV_PATH="$OUT_ROOT/cifar10_wp_hiddenprobe_probe_sweep_s0.csv"

mkdir -p "$OUT_ROOT/logs"

# Build a real CSV from completed summary.json files only.
# Rebuilding from summaries makes interruption/restart safe; replace atomically.
write_csv() {
  python - "$OUT_ROOT" "$CSV_PATH" <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
destination = Path(sys.argv[2])
probe_counts = (8, 16, 32, 64, 128, 256)
columns = (
    "n_probes", "seed", "best_val_tau", "test_tau",
    "test_mse_x1e5", "test_mae", "best_epoch", "best_step",
    "queries", "trainable_params", "lr", "probe_lr",
)
rows = []
for q in probe_counts:
    exp = f"hiddenprobe_cifar10_wp_Q{q}_s0"
    summary_path = root / exp / "summary.json"
    if not summary_path.is_file():
        continue

    summary = json.loads(summary_path.read_text())
    required = (
        "best_val_tau", "final_test_tau", "final_test_accmse_x1e5",
        "final_test_accmae", "best_epoch", "best_step", "queries",
        "params", "lr", "probe_lr",
    )
    missing = [key for key in required if key not in summary]
    if missing:
        raise RuntimeError(
            f"{summary_path}: incomplete summary (missing {missing}); "
            "refusing to treat this experiment as completed"
        )
    if (summary.get("exp") != exp or summary.get("seed") != 0
            or int(summary["queries"]) != q):
        raise RuntimeError(
            f"{summary_path}: exp/seed/query mismatch; not reusing this result"
        )
    rows.append({
        "n_probes": q,
        "seed": 0,
        "best_val_tau": summary["best_val_tau"],
        "test_tau": summary["final_test_tau"],
        "test_mse_x1e5": summary["final_test_accmse_x1e5"],
        "test_mae": summary["final_test_accmae"],
        "best_epoch": summary["best_epoch"],
        "best_step": summary["best_step"],
        "queries": summary["queries"],
        "trainable_params": summary["params"],
        "lr": summary["lr"],
        "probe_lr": summary["probe_lr"],
    })

temporary = destination.with_suffix(destination.suffix + ".tmp")
with temporary.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
os.replace(temporary, destination)
print(f"[CSV] {len(rows)}/6 experiments: {destination}", flush=True)
PY
}

# The dataset/cache is shared by all six runs. Prepare only if it is missing.
CACHE_READY=true
[[ -s "$SPLITS" ]] || CACHE_READY=false
for split in train val test; do
  [[ -s "$CNN_CACHE/cnn_cache_${split}.pt" ]] || CACHE_READY=false
done
if [[ "$CACHE_READY" != true ]]; then
  echo "[DATA] Missing WP cache/splits; running setup once."
  bash scripts/setup_data/regression_cifar10_wp.sh
fi
[[ -s "$SPLITS" ]] || { echo "Missing split: $SPLITS"; exit 2; }
for split in train val test; do
  [[ -s "$CNN_CACHE/cnn_cache_${split}.pt" ]] || {
    echo "Missing CNN cache: $CNN_CACHE/cnn_cache_${split}.pt"
    exit 2
  }
done

# Rebuild immediately in case earlier invocations already completed some Q values.
write_csv

for Q in "${PROBES[@]}"; do
  EXP_NAME="hiddenprobe_cifar10_wp_Q${Q}_s${SEED}"
  EXP_DIR="$OUT_ROOT/$EXP_NAME"
  SUMMARY="$EXP_DIR/summary.json"
  STATE="$EXP_DIR/training_state.pt"
  LOG="$OUT_ROOT/logs/${EXP_NAME}.log"

  # Only accept completed summaries that passed the CSV validator above.
  if [[ -f "$SUMMARY" ]]; then
    echo "[SKIP] Q=$Q: complete summary already exists."
    continue
  fi

  # Keep exactly the same Q-dependent learning rates as the canonical WP runner.
  if [[ "$Q" == 64 ]]; then
    LR=2e-4
    PROBE_LR=2e-4
  else
    LR=3e-4
    PROBE_LR=1e-4
  fi

  RESUME_ARGS=()
  if [[ -s "$STATE" ]]; then
    echo "[RESUME] Q=$Q from $STATE"
    RESUME_ARGS=(--resume_training_state "$STATE")
  else
    echo "[START] Q=$Q from scratch"
    # A killed run may have an incomplete first epoch, but no resumable state.
    # Avoid appending its partial validation trajectory to the new run.
    if [[ -s "$EXP_DIR/log.csv" ]]; then
      mv "$EXP_DIR/log.csv" "$EXP_DIR/log.interrupted.$(date +%Y%m%d_%H%M%S).csv"
    fi
  fi

  CMD=(
    python main.py
    --method hiddenprobe
    --task regression
    --dataset cifar10_wp
    --zoo wp
    --cnn_cache "$CNN_CACHE"
    --splits "$SPLITS"
    --hidden_mode on
    --probe_sharing shared
    --assert_canonical 1
    --target_space raw
    --gen_type deep_linear_5
    --adapter_preset compact
    --interaction_rank 32
    --mixer_hidden 256
    --hidden_dim 0
    --n_probes "$Q"
    --lr "$LR"
    --probe_lr "$PROBE_LR"
    --batch_size 32
    --rank_loss_w 0.0
    --warmup 0
    --grad_clip 0
    --scheduler plateau
    --plateau_factor 0.5
    --plateau_patience 4
    --plateau_min_lr 3e-5
    --probe_min_lr 3e-6
    --epochs 18
    --sched_total_epochs 30
    --n_train 0
    --eval_every 1500
    --eval_cnn_bs "$EVAL_CNN_BS"
    --seed "$SEED"
    --exp_name "$EXP_NAME"
    --out_dir "$EXP_DIR"
    "${RESUME_ARGS[@]}"
  )

  echo "[RUN] Q=$Q seed=$SEED; log=$LOG"
  # pipefail ensures a failed trainer stops the sweep. Completed Q rows
  # remain available in the CSV and will be skipped on the next invocation.
  "${CMD[@]}" 2>&1 | tee -a "$LOG"

  [[ -s "$SUMMARY" ]] || {
    echo "ERROR: Q=$Q exited without producing $SUMMARY"
    exit 1
  }
  write_csv
done

write_csv
echo "[DONE] Probe-count sweep CSV: $CSV_PATH"
