#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

bash scripts/setup_data/regression_fmnist.sh

MAIN_PY="${MAIN_PY:-main.py}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
N_PROBES="${1:-128}"
NUM_SEEDS=5
OUT_ROOT="${OUT_DIR:-checkpoints}"

ZOO_DIR="${ZOO_DIR:-$DATA_ROOT/regression/fmnist}"
SPLIT="$ZOO_DIR/split.csv"
[[ -s "$SPLIT" ]] || { echo "missing official NFN split file: $SPLIT"; exit 2; }
SPLIT_FLAG=(--zoo_split "$SPLIT")

SUMMARIES=()
for ((SEED=0; SEED<NUM_SEEDS; SEED++)); do
  EXP_NAME="hiddenprobe_fmnist_Q${N_PROBES}_s${SEED}"
  SEED_OUT_DIR="$OUT_ROOT/$EXP_NAME"

  python "${MAIN_PY}" \
    --method hiddenprobe --task regression --dataset fmnist \
    --zoo fmnist_gs --gen_type deep_linear_6 --models_c_in 1 \
    --zoo_data_dir "$ZOO_DIR" ${SPLIT_FLAG[@]+"${SPLIT_FLAG[@]}"} \
    --hidden_mode on --probe_sharing shared --assert_canonical 1 --target_space raw \
    --adapter_preset compact --interaction_rank 48 --mixer_hidden 256 --hidden_dim 0 \
    --n_probes "$N_PROBES" --lr 3e-4 --probe_lr 6e-4 --batch_size 32 --rank_loss_w 0.1 --weight_decay 0.0 \
    --scheduler plateau --plateau_factor 0.7 --plateau_patience 4 --plateau_min_lr 3e-5 \
    --epochs 150 --eval_every 500 --val_subset 1485 --eval_cnn_bs 256 \
    --seed "$SEED" --exp_name "$EXP_NAME" --out_dir "$SEED_OUT_DIR"

  SUMMARIES+=("$SEED_OUT_DIR/summary.json")
done

python "$SCRIPT_DIR/aggregate_results.py" \
  --dataset "Fashion-MNIST" \
  --model "HiddenProbe" \
  "${SUMMARIES[@]}"
