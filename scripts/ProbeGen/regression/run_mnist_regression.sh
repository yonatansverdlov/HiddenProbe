#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# Prepare the Small CNN Zoo and both split files:
#   data/regression/mnist/split.csv                canonical ProbeGen/NFN split
#   data/regression/mnist/mnist_gs_auto_split.csv HiddenProbe split
bash scripts/setup_data/regression_mnist.sh

MAIN_PY="${MAIN_PY:-main.py}"
N_PROBES="${1:-128}"

# Canonical MNIST regression configuration from inr_classification_branch,
# expressed with the merged repository's current CLI.
EXP_NAME="probegen_mnist_reg_mlp2_out10_dhid351_Q${N_PROBES}_5seeds_60e_cosine_lr7e-4"

python "$MAIN_PY" \
  --method probegen \
  --task regression \
  --dataset mnist \
  --exp_name "$EXP_NAME" \
  --seed 0 \
  --num_seeds 5 \
  --epochs 60 \
  --batch_size 64 \
  --n_probes "$N_PROBES" \
  --d_hid 351 \
  --mixer_n_layers 6 \
  --gen_type deep_linear_6 \
  --gen_latent_z 32 \
  --generator_width 16 \
  --per_probe_mlp mlp2 \
  --per_probe_mlp_width 270 \
  --per_probe_out_dim 10 \
  --per_probe_init standard \
  --r_per_hidden 2 \
  --rank 8 \
  --scheduler cosine \
  --plateau_monitor val_tau \
  --lr 7e-4 \
  --probe_lr 7e-4 \
  --weight_decay 0.0 \
  --eval_every 500 \
  --n_workers 0 \
  --device cuda
