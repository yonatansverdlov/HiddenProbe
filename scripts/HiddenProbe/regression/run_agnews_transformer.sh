#!/bin/bash
# HiddenProbe on AGNews-Transformers: 256 shared probes (G3, encoder route) + channel-trajectory readout.
# Usage (from the repository root):
#   bash scripts/HiddenProbe/regression/run_agnews_transformer.sh [SEED]         # SEED defaults to 0
# Threshold protocol (retrain on train/val runs with acc >= CUT; the evaluator scores the acc >= CUT test subset):
#   CUT=0.8 bash scripts/HiddenProbe/regression/run_agnews_transformer.sh [SEED]
# Data: run scripts/setup_data/regression_agnews_transformer.sh first (or set DATA_ROOT to your copy).
set -euo pipefail

MAIN_PY="${MAIN_PY:-main.py}"
DATA_ROOT="${DATA_ROOT:-data}"
SEED="${1:-0}"
CUT="${CUT:-0}"

CUT_FLAG=""; CUT_TAG=""
if [[ "$CUT" != "0" ]]; then CUT_FLAG="--cut_off $CUT"; CUT_TAG="_cut${CUT}"; fi

RUNS="checkpoints/hiddenprobe_agnews_transformer${CUT_TAG}_s${SEED}"

python "${MAIN_PY}" transformer train \
  --dataset agnews --generator g3 --n_classes 4 --ffn 384 \
  --n_probes 256 --readout multi --readout_arch channel_trajectory $CUT_FLAG \
  --pred_lr 5e-4 --gen_lr 1e-3 --scheduler plateau --plateau_patience 5 --plateau_factor 0.5 --warmup 0 \
  --weight_decay 1e-3 --dropout 0.1 \
  --max_updates 40000 --eval_every 1000 --micro 32 --seed "$SEED" --exact_accum \
  --data_root "$DATA_ROOT" --runs_dir "$RUNS"

# Test Kendall tau of the best-validation checkpoint (pass several seed dirs at once for a mean +/- std).
python -m models.transformer.evaluate "$RUNS"
