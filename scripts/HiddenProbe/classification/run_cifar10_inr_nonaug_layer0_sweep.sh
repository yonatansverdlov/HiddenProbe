#!/usr/bin/env bash
set -euo pipefail

# HiddenProbe INR layer ablation: CIFAR-10 non-augmented, hidden layer 0 + output.
# Sweep: 24 plateau + 3 cosine configs, seed 0, 20 epochs.
# Final: 5 seeds, 60 epochs, best configuration selected on validation only.
#
# Usage: bash scripts/HiddenProbe/classification/run_cifar10_inr_nonaug_layer0_sweep.sh
#        bash scripts/HiddenProbe/classification/run_cifar10_inr_nonaug_layer0_sweep.sh --dry-run
# Optional: SKIP_DATA_SETUP=1, SWEEP_EPOCHS=20, FINAL_EPOCHS=60, NUM_SEEDS=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# A dry run should be usable without downloading any dataset.
if [[ "${SKIP_DATA_SETUP:-0}" != "1" && " $* " != *" --dry-run "* ]]; then
  bash scripts/setup_data/classification_cifar10.sh
fi

"${PYTHON_BIN:-python}" scripts/HiddenProbe/classification/hidden_layer_ablation_sweep.py \
  --dataset cifar10 \
  --layer-id 0 \
  "$@"
