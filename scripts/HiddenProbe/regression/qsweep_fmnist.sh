#!/bin/bash
# HiddenProbe: Fashion-MNIST-GS accuracy regression -- per-probe-count winning configurations
# Runs main.py once per probe count with THAT probe count's winning configuration (one case per Q),
# so the whole probe-count curve is reproducible. Output: checkpoints/hiddenprobe_fmnist_Q<Q>_s<seed>/
# Usage: bash scripts/HiddenProbe/regression/qsweep_fmnist.sh [SEED]      (default 0; DRYRUN=1 prints the commands instead of running them)
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

ZOO_DIR="${ZOO_DIR:-$DATA_ROOT/regression/fmnist}"        # weights.npy / metrics.csv.gz / layout.csv (scripts/setup_data/regression_fmnist.sh)
# No split csv is shipped for this zoo: the loader generates the seed-0 permutation deterministically
# (written to $ZOO_DIR/fashion_mnist_split.csv on first use and reused afterwards).
SPLIT_FLAG=()

for Q in 16 32 64 128 256; do
  # adapter capacity (preset / interaction rank / mixer width), learning rates and rank-loss weight per probe count
  case "$Q" in
    16)  PRESET=compact; IR=48; MIX=256; LR=5e-4; PROBE_LR=1e-3; RANK_LOSS_W=0.0 ;;
    32)  PRESET=expressive; IR=96; MIX=384; LR=3e-4; PROBE_LR=6e-4; RANK_LOSS_W=0.0 ;;
    64)  PRESET=compact; IR=48; MIX=256; LR=3e-4; PROBE_LR=6e-4; RANK_LOSS_W=0.0 ;;
    128)  PRESET=compact; IR=48; MIX=256; LR=3e-4; PROBE_LR=6e-4; RANK_LOSS_W=0.1 ;;
    256)  PRESET=compact; IR=48; MIX=256; LR=3e-4; PROBE_LR=6e-4; RANK_LOSS_W=0.0 ;;
  esac
  EXP_NAME="hiddenprobe_fmnist_Q${Q}_s${SEED}"
  OUT_DIR="checkpoints/$EXP_NAME"

  run python "${MAIN_PY}" cnn_zoo \
    --zoo fmnist_gs --gen_type deep_linear_6 --models_c_in 1 \
    --zoo_data_dir "$ZOO_DIR" ${SPLIT_FLAG[@]+"${SPLIT_FLAG[@]}"} \
    --hidden_mode on --probe_sharing shared --assert_canonical 1 --target_space raw \
    --adapter_preset "$PRESET" --interaction_rank "$IR" --mixer_hidden "$MIX" --hidden_dim 0 \
    --n_out_probes "$Q" --lr "$LR" --probe_lr "$PROBE_LR" --batch_size 32 --rank_loss_w "$RANK_LOSS_W" --weight_decay 0.0 \
    --scheduler plateau --plateau_factor 0.7 --plateau_patience 4 --plateau_min_lr 3e-5 \
    --epochs 150 --eval_every 500 --val_subset 1485 --eval_cnn_bs 256 \
    --seed "$SEED" --exp_name "$EXP_NAME" --out_dir "$OUT_DIR"
done
