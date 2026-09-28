#!/usr/bin/env bash
# ProbeGen CIFAR10 Wild Park baseline.
# Hyperparameters: jonkahana/ProbeGen, scripts/main_results/
# cifar10_wild_park__ProbeGen_128.sh
# Q is fixed to 128. This merged-repo runner uses seeds 0..4.
# Usage: bash scripts/ProbeGen/regression/run_cifar10_wp_regression.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MAIN_PY="${MAIN_PY:-main.py}"
# All ProbeGen datasets use exactly 128 probes.
if (( $# != 0 )); then
  echo "This runner uses exactly 128 probes; do not pass a probe count." >&2
  exit 2
fi

# Install or verify the canonical dataset and split before training.
bash scripts/setup_data/regression_cifar10_wp.sh

EXP_NAME="probegen_cifar10_wp_Q128_5seeds"
python "$MAIN_PY" \
  --method probegen \
  --task regression \
  --dataset cifar10_wp \
  --exp_name "$EXP_NAME" \
  --seed 0 \
  --num_seeds 5 \
  --n_probes 128 \
  --d_hid 256 \
  --mixer_n_layers 6 \
  --gen_type deep_linear_5 \
  --gen_latent_z 32 \
  --generator_width 16 \
  --per_probe_mlp none \
  --batch_size 32 \
  --lr 3e-4 \
  --scheduler cosine \
  --plateau_monitor val_tau \
  --weight_decay 0.0 \
  --epochs 18 \
  --eval_every 500 \
  --n_workers 0 \
  --device cuda
