#!/bin/bash
# ProbeGen baseline (output-only, --aggregator mlp) on the NON-AUGMENTED CIFAR-10 INR zoo.
# Same probe generator and data as HiddenProbe; Kahana's ProbeGen INR-classification configuration.
# Usage (from the repo root):  bash scripts/ProbeGen/classification/run_cifar10_inr_nonaug.sh [Q] [SEED]
#   Q    = number of learned probes (default 128)
#   SEED = random seed (default 0)
set -euo pipefail

Q="${1:-128}"
SEED="${2:-0}"
DATA_DIR="${DATA_ROOT:-data}/classification/cifar10_inr"

bash scripts/setup_data/classification_cifar10.sh

python main.py inr --method probegen \
  --exp_name="probegen_cifar10_inr_nonaug_Q${Q}_s${SEED}" --seed="$SEED" \
  --dataset=nfn_cifar_inr --dataset_dir="$DATA_DIR" --splits_path=nfn_cifar_split_noaug.json \
  --n_tokens="$Q" --d_hid=256 --mixer_n_layers=6 --aggregator=mlp \
  --gen_type=linear_2_no_acts \
  --batch_size=32 --lr=0.0003 --epochs=30 --eval_every=500 \
  --n_workers=0 --device=cuda
