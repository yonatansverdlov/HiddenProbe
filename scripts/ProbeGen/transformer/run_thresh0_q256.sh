#!/usr/bin/env bash
# Fresh ProbeGen Q256 Transformer training, threshold 0.
# Usage:
#   RUN_TAG=rerun1 bash scripts/ProbeGen/transformer/run_thresh0_q256.sh both
#   RUN_TAG=rerun1 bash scripts/ProbeGen/transformer/run_thresh0_q256.sh mnist
#   RUN_TAG=rerun1 bash scripts/ProbeGen/transformer/run_thresh0_q256.sh agnews 0 1
#
# RUN_TAG is optional. Choose a new tag for a fresh run; existing last.pt
# checkpoints are skipped by the underlying per-dataset runners.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-both}"
if (( $# > 0 )); then shift; fi

case "$DATASET" in
  mnist)
    bash "$SCRIPT_DIR/mnist/run_mnist_transformer_thresh0.sh" "$@"
    ;;
  agnews)
    bash "$SCRIPT_DIR/agnews/run_agnews_transformer_thresh0.sh" "$@"
    ;;
  both)
    bash "$SCRIPT_DIR/mnist/run_mnist_transformer_thresh0.sh" "$@"
    bash "$SCRIPT_DIR/agnews/run_agnews_transformer_thresh0.sh" "$@"
    ;;
  *)
    echo "Usage: $0 {mnist|agnews|both} [seed ...]" >&2
    exit 2
    ;;
esac
