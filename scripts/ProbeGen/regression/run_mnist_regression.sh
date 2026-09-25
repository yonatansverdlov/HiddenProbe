#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

bash scripts/setup_data/regression_mnist.sh

MAIN_PY="${MAIN_PY:-main.py}"


DATASET="mnist"
EXP_NAME="mnist_reg_mlp2_out10_dhid351_FINAL_5SEEDS_60E_cosine_lr7e-4"

python "${MAIN_PY}" \
  --method probegen \
  --task regression \
  --exp_name="${EXP_NAME}" \
  --dataset="${DATASET}" \
  --seed=0 \
  --num_seeds=5 \
  --epochs=60 \
  --batch_size=64 \
  --n_probes=128 \
  --d_hid=351 \
  --mixer_n_layers=6 \
  --gen_type=deep_linear_6 \
  --gen_latent_z=32 \
  --generator_width=16 \
  --per_probe_mlp=mlp2 \
  --per_probe_mlp_width=256 \
  --per_probe_out_dim=10 \
  --r_per_hidden=2 \
  --scheduler=cosine \
  --lr=7e-4 \
  --weight_decay=0.0 \
  --eval_every=500 \
  --n_workers=0 \
  --device=cuda
