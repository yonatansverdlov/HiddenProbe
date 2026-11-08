#!/usr/bin/env bash
set -euo pipefail
# MNIST-Transformers: selected Block 1 + Final LayerNorm + logits.
# Sweep: 27 configs selected by validation Kendall tau; seed 0.
# Final: selected config, five seeds (0..4), mean +/- standard error.
# Defaults: 40000 optimizer updates in each sweep and final run.
# Usage: bash scripts/HiddenProbe/transformer/run_mnist_transformer_block1_sweep.sh [--dry-run|--only-sweep]
# Options: SWEEP_UPDATES, FINAL_UPDATES, NUM_SEEDS, SKIP_DATA_SETUP, DATA_ROOT.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"

if [[ "${SKIP_DATA_SETUP:-0}" != "1" && " $* " != *" --dry-run "* ]]; then
  bash scripts/setup_data/transformer_mnist.sh
  "${PYTHON_BIN:-python}" main.py transformer cache \
    --dataset mnist --seed 0 --cut_off 0 --data_root "$DATA_ROOT"
fi

"${PYTHON_BIN:-python}" scripts/HiddenProbe/transformer/hidden_layer_ablation_sweep.py \
  --dataset mnist --block 1 --data-root "$DATA_ROOT" "$@"
