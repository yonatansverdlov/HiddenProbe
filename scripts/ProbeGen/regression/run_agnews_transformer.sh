#!/bin/bash
# ProbeGen baseline on AGNews-Transformers: the same 256 shared probes (G3, encoder route), output-only
# readout (`rout`: the probe-generator reads ONLY the target's logits, no hidden responses). Readout FFN is
# auto-fit to the parameter cap (--ffn 0).
# Usage (from the repository root):
#   bash scripts/ProbeGen/regression/run_agnews_transformer.sh [SEED]           # SEED defaults to 0
# Threshold protocol: CUT=0.8 bash scripts/ProbeGen/regression/run_agnews_transformer.sh [SEED]
# Data: run scripts/setup_data/regression_agnews_transformer.sh first (or set DATA_ROOT to your copy).
set -euo pipefail

MAIN_PY="${MAIN_PY:-main.py}"
DATA_ROOT="${DATA_ROOT:-data}"
SEED="${1:-0}"
CUT="${CUT:-0}"

CUT_FLAG=""; CUT_TAG=""
if [[ "$CUT" != "0" ]]; then CUT_FLAG="--cut_off $CUT"; CUT_TAG="_cut${CUT}"; fi

RUNS="checkpoints/probegen_agnews_transformer${CUT_TAG}_s${SEED}"

python "${MAIN_PY}" transformer train \
  --dataset agnews --generator g3 --n_classes 4 --ffn 0 \
  --n_probes 256 --readout multi --pma_seeds 4 --readout_arch rout $CUT_FLAG \
  --pred_lr 5e-4 --gen_lr 5e-4 --scheduler cosine --plateau_patience 0 --plateau_factor 0 \
  --weight_decay 1e-3 --dropout 0.1 --warmup 0 \
  --max_updates 40000 --eval_every 1000 --micro 32 --seed "$SEED" \
  --data_root "$DATA_ROOT" --runs_dir "$RUNS"

# Test Kendall tau of the best-validation checkpoint (pass several seed dirs at once for a mean +/- std).
python -m models.transformer.evaluate "$RUNS"
