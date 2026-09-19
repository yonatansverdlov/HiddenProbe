#!/bin/bash
# HiddenProbe on the NON-AUGMENTED CIFAR-10 INR zoo (NFN siren_cifar_wts, one SIREN per image).
# Usage (from the repo root):  bash scripts/HiddenProbe/classification/run_cifar10_inr_nonaug.sh [Q] [SEED]
#   Q    = number of learned probes (default 128)
#   SEED = random seed (default 0)
set -euo pipefail

Q="${1:-128}"
SEED="${2:-0}"
DATA_DIR="${DATA_ROOT:-data}/classification/cifar10_inr"

bash scripts/setup_data/classification_cifar10.sh

python main.py inr --method hiddenprobe \
  --dataset nfn_cifar_inr --dataset_dir "$DATA_DIR" --splits nfn_cifar_split_noaug.json \
  --L 2 --H 32 --out_dim 3 --n_classes 10 --models_c_in 2 \
  --gen_type linear_2_no_acts --gen_latent_z 32 --generator_width 16 --n_probes "$Q" --domain_tanh 1 \
  --head set_transformer --d 120 --nenc 2 --nheads 8 \
  --use_post_act 1 --siren_w0 30 --use_neuron_stats 1 --ema_decay 0.999 \
  --lr 2e-4 --probe_lr 2e-3 --batch_size 32 --warmup 300 --dropout 0.1 --head_wd 0.1 \
  --scheduler plateau --plateau_factor 0.5 --plateau_patience 3 --plateau_min_lr 1e-6 \
  --epochs 30 --eval_every 500 --n_train 0 --seed "$SEED" \
  --runs_dir "checkpoints/cifar10_inr_nonaug" --exp_name "hiddenprobe_cifar10_inr_nonaug_Q${Q}_s${SEED}"
