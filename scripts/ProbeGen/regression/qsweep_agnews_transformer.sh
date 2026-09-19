#!/bin/bash
# ProbeGen baseline probe-count (Q) sweep on AGNews-Transformers: for every Q the output-only readout (`rout`,
# logits only, no hidden responses) with the same shared G3 probes (encoder route), cosine schedule, readout
# FFN auto-fit to the parameter cap (--ffn 0) -- i.e. run_agnews_transformer.sh (ProbeGen) with --n_probes Q.
# Usage (from the repository root):
#   bash scripts/ProbeGen/regression/qsweep_agnews_transformer.sh [SEED]        # SEED defaults to 0
#   DRYRUN=1 bash scripts/ProbeGen/regression/qsweep_agnews_transformer.sh      # only print the commands
# Each run is followed by `python -m models.transformer.evaluate <run_dir>` (test Kendall tau of the best-val checkpoint).
# Data: run scripts/setup_data/regression_agnews_transformer.sh first (or set DATA_ROOT to your copy).
set -euo pipefail

MAIN_PY="${MAIN_PY:-main.py}"
DATA_ROOT="${DATA_ROOT:-data}"
SEED="${1:-0}"
DRYRUN="${DRYRUN:-0}"

run() {                                   # echo the command; execute it unless DRYRUN=1
  echo "+ $*"
  if [[ "$DRYRUN" != "1" ]]; then "$@"; fi
}

for Q in 16 32 64 128 256; do
  RUNS="checkpoints/probegen_agnews_transformer_Q${Q}_s${SEED}"
  run python "${MAIN_PY}" transformer train \
    --dataset agnews --generator g3 --n_classes 4 --ffn 0 \
    --n_probes "$Q" --readout multi --pma_seeds 4 --readout_arch rout \
    --pred_lr 5e-4 --gen_lr 5e-4 --scheduler cosine --plateau_patience 0 --plateau_factor 0 \
    --weight_decay 1e-3 --dropout 0.1 --warmup 0 \
    --max_updates 40000 --eval_every 1000 --micro 32 --seed "$SEED" \
    --data_root "$DATA_ROOT" --runs_dir "$RUNS"
  run python -m models.transformer.evaluate "$RUNS"
done
