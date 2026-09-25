#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# Make the run self-contained: install/verify MNIST INR data first.
bash scripts/setup_data/classification_mnist.sh

# Preserve the environment used by the original MNIST HiddenProbe run.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate probegen
fi

python main.py \
    --method hiddenprobe \
    --task classification \
    --dataset mnist \
    --exp_name mnist_inr_hidden_probe \
    --seed 0 \
    --num_seeds 5 \
    --epochs 30 \
    --batch_size 64 \
    --d_hid 256 \
    --mixer_n_layers 6 \
    --gen_type linear_2_no_acts \
    --gen_latent_z 32 \
    --generator_width 16 \
    --scheduler plateau \
    --plateau_monitor val_acc \
    --plateau_patience 3 \
    --plateau_factor 0.3 \
    --plateau_min_lr 1e-6 \
    --lr 0.0007 \
    --weight_decay 0.0 \
    --eval_every 500 \
    --n_workers 0 \
    --per_probe_mlp mlp2 \
    --per_probe_mlp_width 256 \
    --per_probe_out_dim 4 \
    --per_probe_init standard \
    --n_probes 128 \
    --r_per_hidden 2 \
    --rank 8 \
    --device cuda
