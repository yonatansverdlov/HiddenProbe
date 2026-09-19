#!/bin/bash
# HiddenProbe on the AUGMENTED CIFAR-10 INR zoo -- per-probe-count winning configurations
# Runs main.py once per probe count with THAT probe count's winning configuration (one case per Q).
# Usage (from the repo root):  bash scripts/HiddenProbe/classification/qsweep_cifar10_inr_aug.sh [SEED]
#   SEED = random seed (default 0);  DRYRUN=1 prints the commands instead of running them
set -euo pipefail

SEED="${1:-0}"
DRYRUN="${DRYRUN:-0}"
DATA_DIR="${DATA_ROOT:-data}/classification/cifar10_inr"

run() {
  if [[ "$DRYRUN" == "1" ]]; then printf '%s ' "$@"; printf '\n'; else "$@"; fi
}

[[ "$DRYRUN" == "1" ]] || bash scripts/setup_data/classification_cifar10.sh

for Q in 16 32 64 128 256; do
  # set-transformer head width / depth, learning rates, input features and schedule per probe count
  case "$Q" in
    16)  D=120; NENC=3; LR=2e-4; PROBE_LR=2e-3; FEAT=(--feat_fourier 3 --feat_order 16); SCHED=(--scheduler plateau --plateau_factor 0.5 --plateau_patience 8 --plateau_min_lr 1e-6) ;;
    32)  D=112; NENC=3; LR=4e-4; PROBE_LR=4e-3; FEAT=(--feat_fourier 3 --feat_order 16); SCHED=(--scheduler cosine --plateau_min_lr 1e-5) ;;
    64)  D=112; NENC=3; LR=4e-4; PROBE_LR=4e-3; FEAT=(--feat_fourier 3 --feat_order 16); SCHED=(--scheduler cosine --plateau_min_lr 1e-5) ;;
    128)  D=112; NENC=3; LR=4e-4; PROBE_LR=4e-3; FEAT=(); SCHED=(--scheduler cosine --plateau_min_lr 1e-5) ;;
    256)  D=104; NENC=3; LR=4e-4; PROBE_LR=4e-3; FEAT=(); SCHED=(--scheduler cosine --plateau_min_lr 1e-5) ;;
  esac
  EXP_NAME="hiddenprobe_cifar10_inr_aug_Q${Q}_s${SEED}"

  run python main.py inr --method hiddenprobe \
    --dataset nfn_cifar_inr --dataset_dir "$DATA_DIR" --splits nfn_cifar_split.json \
    --L 2 --H 32 --out_dim 3 --n_classes 10 --models_c_in 2 \
    --gen_type linear_2_no_acts --gen_latent_z 32 --generator_width 16 --n_probes "$Q" --domain_tanh 1 \
    --head set_transformer --d "$D" --nenc "$NENC" --nheads 8 \
    --use_post_act 1 --siren_w0 30 --use_neuron_stats 1 --ema_decay 0.999 ${FEAT[@]+"${FEAT[@]}"} \
    --lr "$LR" --probe_lr "$PROBE_LR" --batch_size 32 --warmup 300 --dropout 0.1 --head_wd 0.1 \
    "${SCHED[@]}" \
    --epochs 12 --eval_every 500 --n_train 0 --seed "$SEED" \
    --runs_dir "checkpoints/cifar10_inr_aug" --exp_name "$EXP_NAME"
done
