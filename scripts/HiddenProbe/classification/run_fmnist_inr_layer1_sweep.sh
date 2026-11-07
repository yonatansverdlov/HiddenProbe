#!/usr/bin/env bash
set -euo pipefail

# HiddenProbe INR layer ablation: Fashion-MNIST, hidden layer 1 + output.
# Sweep: 24 plateau + 3 cosine configs, seed 0, 20 epochs.
# Final: 5 seeds, 30 epochs, best configuration selected on validation only.
#
# Usage: bash scripts/HiddenProbe/classification/run_fmnist_inr_layer1_sweep.sh
#        bash scripts/HiddenProbe/classification/run_fmnist_inr_layer1_sweep.sh --dry-run
# Optional: SKIP_DATA_SETUP=1, SWEEP_EPOCHS=20, FINAL_EPOCHS=30, NUM_SEEDS=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# A dry run should be usable without downloading any dataset.
if [[ "${SKIP_DATA_SETUP:-0}" != "1" && " $* " != *" --dry-run "* ]]; then
  bash scripts/setup_data/classification_fmnist.sh
fi

# The canonical FMNIST trainer uses the probegen conda environment.
if [[ "${SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-probegen}"
fi

"${PYTHON_BIN:-python}" scripts/HiddenProbe/classification/hidden_layer_ablation_sweep.py \
  --dataset fmnist \
  --layer-id 1 \
  "$@"
