#!/bin/bash
# ProbeGen baseline (output-only, --aggregator mlp) on the NON-AUGMENTED CIFAR-10 INR zoo -- all probe counts
# Runs main.py once per probe count with THAT probe count's winning configuration (one case per Q).
# Usage (from the repo root):  bash scripts/ProbeGen/classification/qsweep_cifar10_inr_nonaug.sh [SEED]
#   SEED = random seed (default 0);  DRYRUN=1 prints the commands instead of running them
set -euo pipefail

SEED="${1:-0}"
DRYRUN="${DRYRUN:-0}"
DATA_DIR="${DATA_ROOT:-data}/classification/cifar10_inr"

run() {
  if [[ "$DRYRUN" == "1" ]]; then printf '%s ' "$@"; printf '\n'; else "$@"; fi
}

[[ "$DRYRUN" == "1" ]] || bash scripts/setup_data/classification_cifar10.sh

# Same command as run_cifar10_inr_nonaug.sh (Kahana's ProbeGen INR-classification configuration), one run per probe count.
for Q in 16 32 64 128 256; do
  EXP_NAME="probegen_cifar10_inr_nonaug_Q${Q}_s${SEED}"

  run python main.py inr --method probegen \
    --exp_name="$EXP_NAME" --seed="$SEED" \
    --dataset=nfn_cifar_inr --dataset_dir="$DATA_DIR" --splits_path=nfn_cifar_split_noaug.json \
    --n_tokens="$Q" --d_hid=256 --mixer_n_layers=6 --aggregator=mlp \
    --gen_type=linear_2_no_acts \
    --batch_size=32 --lr=0.0003 --epochs=30 --eval_every=500 \
    --n_workers=0 --device=cuda
done
