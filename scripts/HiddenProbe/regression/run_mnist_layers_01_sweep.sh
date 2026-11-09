#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ "${SKIP_DATA_SETUP:-0}" != "1" && " $* " != *" --dry-run "* ]]; then
  bash scripts/setup_data/regression_mnist.sh
fi
"${PYTHON_BIN:-python}" scripts/HiddenProbe/regression/mnist_hidden_layer_ablation_sweep.py \
  --layers 0 1 "$@"
