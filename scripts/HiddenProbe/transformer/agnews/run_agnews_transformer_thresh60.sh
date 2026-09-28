#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-data}"
CUT="0.6"
NUM_SEEDS=5

# Threshold-specific selected configuration from the HiddenProbe reference branch.
PRED_LR=2.5e-4
GEN_LR=1e-3
DROPOUT=0.1

bash scripts/setup_data/regression_agnews_transformer.sh

# Transformer-NFN threshold protocol:
# filter epoch-75 models by absolute accuracy first, then make a fresh
# deterministic 70/15/15 split for this threshold.
python main.py transformer cache \
  --dataset agnews \
  --seed 0 \
  --cut_off "$CUT" \
  --data_root "$DATA_ROOT"

RUN_DIRS=()
for ((SEED=0; SEED<NUM_SEEDS; SEED++)); do
  RUNS="checkpoints/hiddenprobe_agnews_transformer_thresh60_s${SEED}"

  if [[ ! -s "$RUNS/last.pt" ]]; then
    python main.py transformer train \
      --dataset agnews \
      --generator g3 \
      --n_classes 4 \
      --ffn 384 \
      --n_probes 256 \
      --readout multi \
      --readout_arch channel_trajectory \
      --cut_off "$CUT" \
      --pred_lr "$PRED_LR" \
      --gen_lr "$GEN_LR" \
      --scheduler plateau \
      --plateau_patience 5 \
      --plateau_factor 0.5 \
      --warmup 0 \
      --weight_decay 1e-3 \
      --dropout "$DROPOUT" \
      --max_updates 40000 \
      --eval_every 1000 \
      --micro 32 \
      --seed "$SEED" \
      --exact_accum \
      --data_root "$DATA_ROOT" \
      --runs_dir "$RUNS"
  fi

  RUN_DIRS+=("$RUNS")
done

python -m models.transformer.evaluate "${RUN_DIRS[@]}"
