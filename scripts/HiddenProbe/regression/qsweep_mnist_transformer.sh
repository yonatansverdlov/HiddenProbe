#!/bin/bash
# HiddenProbe probe-count (Q) sweep on MNIST-Transformers: the per-Q winning configurations.
#   Q = 16, 32, 64, 128 : r2 cross-layer-fusion readout + token_mlp ("r2tm"), shared G3 probes, cosine schedule,
#                         readout FFN auto-fit to the parameter cap (--ffn 0).
#   Q = 256             : the channel-trajectory readout (Q256-only; same command as run_mnist_transformer.sh).
# Usage (from the repository root):
#   bash scripts/HiddenProbe/regression/qsweep_mnist_transformer.sh [SEED]       # SEED defaults to 0
#   DRYRUN=1 bash scripts/HiddenProbe/regression/qsweep_mnist_transformer.sh     # only print the commands
# Each run is followed by `python -m models.transformer.evaluate <run_dir>` (test Kendall tau of the best-val checkpoint).
# Data: run scripts/setup_data/regression_mnist_transformer.sh first (or set DATA_ROOT to your copy).
set -euo pipefail

MAIN_PY="${MAIN_PY:-main.py}"
DATA_ROOT="${DATA_ROOT:-data}"
SEED="${1:-0}"
DRYRUN="${DRYRUN:-0}"

run() {                                   # echo the command; execute it unless DRYRUN=1
  echo "+ $*"
  if [[ "$DRYRUN" != "1" ]]; then "$@"; fi
}

for Q in 16 32 64 128; do
  RUNS="checkpoints/hiddenprobe_mnist_transformer_Q${Q}_s${SEED}"
  run python "${MAIN_PY}" transformer train \
    --dataset mnist --generator g3 --n_classes 10 --ffn 0 \
    --n_probes "$Q" --readout multi --pma_seeds 4 --readout_arch r2 --token_mlp \
    --pred_lr 5e-4 --gen_lr 5e-4 --scheduler cosine --plateau_patience 0 --plateau_factor 0 \
    --weight_decay 1e-3 --dropout 0.1 --warmup 0 \
    --max_updates 40000 --eval_every 1000 --micro 32 --seed "$SEED" \
    --data_root "$DATA_ROOT" --runs_dir "$RUNS"
  run python -m models.transformer.evaluate "$RUNS"
done

Q=256
RUNS="checkpoints/hiddenprobe_mnist_transformer_Q${Q}_s${SEED}"
run python "${MAIN_PY}" transformer train \
  --dataset mnist --generator g3 --n_classes 10 --ffn 384 \
  --n_probes "$Q" --readout multi --readout_arch channel_trajectory \
  --pred_lr 2.5e-4 --gen_lr 5e-4 --scheduler plateau --plateau_patience 5 --plateau_factor 0.5 --warmup 0 \
  --weight_decay 1e-3 --dropout 0.1 \
  --max_updates 40000 --eval_every 1000 --micro 32 --seed "$SEED" --exact_accum \
  --data_root "$DATA_ROOT" --runs_dir "$RUNS"
run python -m models.transformer.evaluate "$RUNS"
