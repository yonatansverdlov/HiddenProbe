#!/bin/bash
# HiddenProbe: CIFAR10 Wild Park accuracy regression -- per-probe-count winning configurations (canonical WP protocol)
# Runs main.py once per probe count with THAT probe count's winning configuration (one case per Q),
# so the whole probe-count curve is reproducible. Output: checkpoints/hiddenprobe_cifar10_wp_Q<Q>_s<seed>/
# Usage: bash scripts/HiddenProbe/regression/qsweep_cifar10_wp.sh [SEED]      (default 0; DRYRUN=1 prints the commands instead of running them)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MAIN_PY="${MAIN_PY:-main.py}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
SEED="${1:-0}"
DRYRUN="${DRYRUN:-0}"

run() {
  if [[ "$DRYRUN" == "1" ]]; then printf '%s ' "$@"; printf '\n'; else "$@"; fi
}

WP_DIR="${WP_DIR:-$DATA_ROOT/regression/cifar10_wp}"
CNN_CACHE="${CNN_CACHE:-$WP_DIR/wp_cnn_cache}"        # cnn_cache_{train,val,test}.pt (scripts/setup_data/regression_cifar10_wp.sh)
SPLITS="$REPO_ROOT/scripts/setup_data/splits/cnn_park_splits.json"
if [[ "$DRYRUN" != "1" ]]; then
  for s in train val test; do
    [[ -s "$CNN_CACHE/cnn_cache_$s.pt" ]] || { echo "missing $CNN_CACHE/cnn_cache_$s.pt — run scripts/setup_data/regression_cifar10_wp.sh"; exit 2; }
  done
fi

for Q in 16 32 64 128; do
  # optimisation schedule per probe count (all share the canonical box: compact / ir32 / mix256 / deep_linear_5).
  # 16 and 32 probes keep the trainer defaults for the minimum learning rates (--plateau_min_lr 1e-5, no probe floor);
  # 64 and 128 probes floor both learning rates.
  case "$Q" in
    16)  LR=3e-4; PROBE_LR=1e-3; EPOCHS=8;  SCHED=(--scheduler cosine  --plateau_factor 0.3 --plateau_patience 4 --plateau_min_lr 1e-5) ;;
    32)  LR=3e-4; PROBE_LR=3e-4; EPOCHS=12; SCHED=(--scheduler plateau --plateau_factor 0.3 --plateau_patience 3 --plateau_min_lr 1e-5) ;;
    64)  LR=2e-4; PROBE_LR=2e-4; EPOCHS=18; SCHED=(--scheduler plateau --plateau_factor 0.5 --plateau_patience 4 --plateau_min_lr 3e-5 --probe_min_lr 3e-6) ;;
    128) LR=3e-4; PROBE_LR=1e-4; EPOCHS=18; SCHED=(--scheduler plateau --plateau_factor 0.5 --plateau_patience 4 --plateau_min_lr 3e-5 --probe_min_lr 3e-6) ;;
  esac
  EXP_NAME="hiddenprobe_cifar10_wp_Q${Q}_s${SEED}"
  OUT_DIR="checkpoints/$EXP_NAME"

  run python "${MAIN_PY}" cnn_zoo \
    --zoo wp --cnn_cache "$CNN_CACHE" --splits "$SPLITS" \
    --hidden_mode on --probe_sharing shared --assert_canonical 1 --target_space raw \
    --gen_type deep_linear_5 --adapter_preset compact --interaction_rank 32 --mixer_hidden 256 --hidden_dim 0 \
    --n_out_probes "$Q" --lr "$LR" --probe_lr "$PROBE_LR" --batch_size 32 --rank_loss_w 0.0 --warmup 0 --grad_clip 0 \
    "${SCHED[@]}" \
    --epochs "$EPOCHS" --sched_total_epochs 30 --n_train 0 --eval_every 1500 --val_subset 1485 --eval_cnn_bs 256 \
    --seed "$SEED" --exp_name "$EXP_NAME" --out_dir "$OUT_DIR" --dump_preds "$OUT_DIR/preds.npz"
done
