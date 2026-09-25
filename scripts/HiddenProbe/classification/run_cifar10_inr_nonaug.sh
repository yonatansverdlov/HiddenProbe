#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

bash scripts/setup_data/classification_cifar10.sh

Q="${1:-128}"
NUM_SEEDS=5
RUNS_DIR="checkpoints/cifar10_inr_nonaug"
SUMMARIES=()

for ((SEED=0; SEED<NUM_SEEDS; SEED++)); do
  EXP_NAME="hiddenprobe_cifar10_inr_nonaug_Q${Q}_s${SEED}"
  SUMMARY="$RUNS_DIR/$EXP_NAME/summary.json"

  python "$SCRIPT_DIR/recover_cifar_summary.py" \
    --run_dir "$RUNS_DIR/$EXP_NAME" \
    --exp_name "$EXP_NAME" \
    --seed "$SEED"

  if [[ ! -s "$SUMMARY" ]]; then
    python main.py \
      --method hiddenprobe \
      --task classification \
      --dataset cifar10 \
      --gen_type linear_2_no_acts --gen_latent_z 32 --generator_width 16 --n_probes "$Q" --domain_tanh 1 \
      --head set_transformer --d 120 --nenc 2 --nheads 8 \
      --ema_decay 0.999 \
      --lr 2e-4 --probe_lr 7e-4 --batch_size 32 --warmup 300 --dropout 0.1 --head_wd 0.1 \
      --scheduler plateau --plateau_factor 0.7 --plateau_patience 5 --plateau_min_lr 1e-6 \
      --epochs 60 --eval_every 500 \
      --seed "$SEED" \
      --runs_dir "$RUNS_DIR" \
      --exp_name "$EXP_NAME"
  fi

  SUMMARIES+=("$SUMMARY")
done

python "$SCRIPT_DIR/aggregate_results.py" \
  --dataset "CIFAR-10" \
  --model "HiddenProbe" \
  "${SUMMARIES[@]}"
