#!/usr/bin/env bash
# ProbeGen AGNEWS-Transformer baseline, threshold 0%.
# Q=256, output-only (rout), auto-fit FFN, cosine schedule.
# Default seeds: 0 1 2 3 4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

MAIN_PY="${MAIN_PY:-main.py}"
DATA_ROOT="${DATA_ROOT:-data}"
CUT="0"
CUT_PCT=0

SEEDS=("$@")
if (( ${#SEEDS[@]} == 0 )); then SEEDS=(0 1 2 3 4); fi

bash scripts/setup_data/transformer_agnews.sh
python "$MAIN_PY" transformer cache \
  --dataset agnews --seed 0 --cut_off "$CUT" --data_root "$DATA_ROOT"

# Optional namespace for a fresh rerun without overwriting existing checkpoints.
# Example: RUN_TAG=rerun1 bash scripts/ProbeGen/transformer/agnews/run_agnews_transformer_thresh0.sh
RUN_TAG="${RUN_TAG:-}"
if [[ -n "$RUN_TAG" && ! "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RUN_TAG must contain only letters, digits, underscores, or hyphens" >&2
  exit 2
fi
PREFIX="checkpoints/probegen_agnews_transformer_Q256_thresh${CUT_PCT}${RUN_TAG:+_${RUN_TAG}}"
RUN_DIRS=()
for SEED in "${SEEDS[@]}"; do
  RUNS="${PREFIX}_s${SEED}"
  if [[ ! -s "$RUNS/last.pt" ]]; then
    python "$MAIN_PY" transformer train \
      --method probegen \
      --dataset agnews --generator g3 --n_classes 4 --ffn 0 \
      --n_probes 256 --readout multi --pma_seeds 4 --readout_arch rout \
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
