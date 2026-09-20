#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

bash scripts/setup_data/classification_cifar10.sh

Q="${1:-128}"
NUM_SEEDS=5
RUNS_DIR="checkpoints/cifar10_inr_aug"
SUMMARIES=()

for ((SEED=0; SEED<NUM_SEEDS; SEED++)); do
  EXP_NAME="hiddenprobe_cifar10_inr_aug_Q${Q}_s${SEED}"
  SUMMARY="$RUNS_DIR/$EXP_NAME/summary.json"

  if [[ ! -s "$SUMMARY" ]]; then
    python main.py \
      --method hiddenprobe \
      --task classification \
      --dataset cifar10_aug \
      --gen_type linear_2_no_acts --gen_latent_z 32 --generator_width 16 --n_probes "$Q" --domain_tanh 1 \
      --head set_transformer --d 112 --nenc 3 --nheads 8 \
      --use_post_act 1 --siren_w0 30 --use_neuron_stats 1 --ema_decay 0.999 \
      --lr 4e-4 --probe_lr 4e-3 --batch_size 32 --warmup 300 --dropout 0.1 --head_wd 0.1 \
      --scheduler cosine --plateau_min_lr 1e-5 \
      --epochs 12 --eval_every 500 --n_train 0 \
      --seed "$SEED" \
      --runs_dir "$RUNS_DIR" \
      --exp_name "$EXP_NAME"
  fi

  SUMMARIES+=("$SUMMARY")
done

python "$SCRIPT_DIR/aggregate_results.py" \
  --dataset "CIFAR-10 Augmented" \
  --model "HiddenProbe" \
  "${SUMMARIES[@]}"
