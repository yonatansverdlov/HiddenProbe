#!/bin/bash
# HiddenProbe: CIFAR10 Wild Park accuracy regression (canonical WP protocol)
# Usage: bash scripts/HiddenProbe/regression/run_cifar10_wp_regression.sh [N_PROBES]      (default 128; SEED=<n> to change the seed, default 0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MAIN_PY="${MAIN_PY:-main.py}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
N_PROBES="${1:-128}"
SEED="${SEED:-0}"

WP_DIR="${WP_DIR:-$DATA_ROOT/regression/cifar10_wp}"
CNN_CACHE="${CNN_CACHE:-$WP_DIR/wp_cnn_cache}"        # cnn_cache_{train,val,test}.pt (scripts/setup_data/regression_cifar10_wp.sh)
SPLITS="$REPO_ROOT/scripts/setup_data/splits/cnn_park_splits.json"
for s in train val test; do
  [[ -s "$CNN_CACHE/cnn_cache_$s.pt" ]] || { echo "missing $CNN_CACHE/cnn_cache_$s.pt — run scripts/setup_data/regression_cifar10_wp.sh"; exit 2; }
done

# learning rates per probe count (64 probes use their own pair; every other count uses the 128-probe pair)
if [[ "$N_PROBES" == "64" ]]; then LR=2e-4; PROBE_LR=2e-4; else LR=3e-4; PROBE_LR=1e-4; fi

EXP_NAME="hiddenprobe_cifar10_wp_Q${N_PROBES}_s${SEED}"
OUT_DIR="${OUT_DIR:-checkpoints/$EXP_NAME}"

python "${MAIN_PY}" cnn_zoo \
  --zoo wp --cnn_cache "$CNN_CACHE" --splits "$SPLITS" \
  --hidden_mode on --probe_sharing shared --assert_canonical 1 --target_space raw \
  --gen_type deep_linear_5 --adapter_preset compact --interaction_rank 32 --mixer_hidden 256 --hidden_dim 0 \
  --n_out_probes "$N_PROBES" --lr "$LR" --probe_lr "$PROBE_LR" --batch_size 32 --rank_loss_w 0.0 --warmup 0 --grad_clip 0 \
  --scheduler plateau --plateau_factor 0.5 --plateau_patience 4 --plateau_min_lr 3e-5 --probe_min_lr 3e-6 \
  --epochs 18 --sched_total_epochs 30 --n_train 0 --eval_every 1500 --val_subset 1485 --eval_cnn_bs 256 \
  --seed "$SEED" --exp_name "$EXP_NAME" --out_dir "$OUT_DIR" --dump_preds "$OUT_DIR/preds.npz"
