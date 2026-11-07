#!/usr/bin/env bash
set -euo pipefail
# MNIST-Transformers (Kendall tau): ProbeGen vs HiddenProbe cost sweep (Q=16,32,64,128,256).
# Run from anywhere. Remaining arguments are forwarded to the Python benchmark.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python measurments/benchmark_other_probe_cost.py --dataset mnist_transformer "$@"
