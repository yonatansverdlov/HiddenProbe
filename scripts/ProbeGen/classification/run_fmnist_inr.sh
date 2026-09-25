#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

bash scripts/setup_data/classification_fmnist.sh

MAIN_PY="${MAIN_PY:-main.py}"

python "$MAIN_PY" \
  --method probegen \
  --task classification \
  --exp_name=ProbeGen_128__seed_1 \
  --seed=1 \
  --dataset=fmnist \
  \
  --n_probes=128 \
  --d_hid=256 \
  --mixer_n_layers=6 \
  \
  --gen_type=linear_2_no_acts \
  \
  --batch_size=32 \
  --lr=0.0003 \
  --epochs=30 \
  --eval_every=500 \
  --n_workers=0 \
  --device=cuda