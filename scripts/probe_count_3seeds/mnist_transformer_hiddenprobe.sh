#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python -u scripts/probe_count_5seeds.py \
  --dataset mnist_transformer \
  --methods hiddenprobe \
  --seeds 0 1 2 \
  --out-root checkpoints/probe_count_3seeds/hiddenprobe \
  --skip-setup
