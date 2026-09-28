#!/usr/bin/env bash
# ProbeGen AGNEWS-Transformer baseline, threshold 20%.
# Q=128, output-only (rout), auto-fit FFN, cosine schedule.
# Default seeds: 0 1 2 3 4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

MAIN_PY="${MAIN_PY:-main.py}"
DATA_ROOT="${DATA_ROOT:-data}"
CUT="0.2"
CUT_PCT=20

SEEDS=("$@")
if (( ${#SEEDS[@]} == 0 )); then SEEDS=(0 1 2 3 4); fi

bash scripts/setup_data/regression_agnews_transformer.sh
python "$MAIN_PY" transformer cache \
  --dataset agnews --seed 0 --cut_off "$CUT" --data_root "$DATA_ROOT"

PREFIX="checkpoints/probegen_agnews_transformer_Q128_thresh${CUT_PCT}"
RUN_DIRS=()
for SEED in "${SEEDS[@]}"; do
  RUNS="${PREFIX}_s${SEED}"
  if [[ ! -s "$RUNS/last.pt" ]]; then
    python "$MAIN_PY" transformer train \
      --dataset agnews --generator g3 --n_classes 4 --ffn 0 \
      --n_probes 128 --readout multi --pma_seeds 4 --readout_arch rout \
      --cut_off "$CUT" \
      --pred_lr 5e-4 --gen_lr 5e-4 --scheduler cosine \
      --plateau_patience 0 --plateau_factor 0 \
      --weight_decay 1e-3 --dropout 0.1 --warmup 0 \
      --max_updates 40000 --eval_every 1000 --micro 32 --seed "$SEED" \
      --data_root "$DATA_ROOT" --runs_dir "$RUNS"
  fi
  RUN_DIRS+=("$RUNS")
done

python -m models.transformer.evaluate "${RUN_DIRS[@]}"
