#!/usr/bin/env bash
set -euo pipefail

# Standalone CIFAR10 non-augmented HiddenProbe sweep. Original run script unchanged.
# 24 configs, seed 9, 15 epochs. Pick best validation config, then 0..4 at 60 epochs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ "${SKIP_DATA_SETUP:-0}" != "1" ]]; then
    bash scripts/setup_data/classification_cifar10.sh
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_PY="${MAIN_PY:-main.py}"
Q="${1:-128}"
RUNS_DIR="${RUNS_DIR:-checkpoints/cifar10_inr_nonaug}"
PREFIX="hiddenprobe_cifar10_inr_nonaug_Q${Q}"
SWEEP_DIR="$RUNS_DIR/sweeps/$PREFIX"
RANKED_CSV="$SWEEP_DIR/ranked_24.csv"
BEST_ENV="$SWEEP_DIR/best.env"
FINAL_CSV="$SWEEP_DIR/final_5seeds.csv"
mkdir -p "$SWEEP_DIR"

LRS=(3e-4 5e-4 7e-4)
PATIENCES=(3 5)
FACTORS=(0.2 0.3 0.5 0.7)
SWEEP_SEED=9
SWEEP_EPOCHS=15
FINAL_EPOCHS=60

# Original CIFAR non-aug architecture: set_transformer d=120, nenc=2, nheads=8.
# main.py pins target INRs to 2 hidden SIREN layers of width 32, RGB output,
# and uses the non-augmented nfn_cifar_split_noaug.json split.
run_one() {
    local exp="$1" seed="$2" epochs="$3" lr="$4" patience="$5" factor="$6"
    "$PYTHON_BIN" "$MAIN_PY" \
        --method hiddenprobe --task classification --dataset cifar10 \
        --gen_type linear_2_no_acts --gen_latent_z 32 --generator_width 16 \
        --n_probes "$Q" --domain_tanh 1 \
        --head set_transformer --d 120 --nenc 2 --nheads 8 \
        --ema_decay 0.999 \
        --lr "$lr" --probe_lr "$lr" \
        --batch_size 32 --warmup 300 --dropout 0.1 --head_wd 0.1 \
        --scheduler plateau --plateau_factor "$factor" \
        --plateau_patience "$patience" --plateau_min_lr 1e-6 \
        --epochs "$epochs" --eval_every 500 \
        --seed "$seed" --runs_dir "$RUNS_DIR" --exp_name "$exp"
}

# CIFAR trainer writes per-run summary.json, not seeds_summary.csv.
summary_complete() {
    "$PYTHON_BIN" - "$1" "$2" "$3" <<'PY'
import json
import math
import sys
from pathlib import Path
path, exp, seed = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
try:
    with path.open() as f:
        s = json.load(f)
    assert s["exp"] == exp and int(s["seed"]) == seed
    assert (path.parent / "best.pt").is_file()
    for key in ("best_val_acc", "best_test_acc", "final_val_acc"):
        x = float(s[key])
        assert math.isfinite(x) and 0 <= x <= 1
except (OSError, ValueError, TypeError, KeyError, AssertionError):
    sys.exit(1)
PY
}

run_if_needed() {
    local exp="$1" seed="$2" epochs="$3" lr="$4" patience="$5" factor="$6"
    local summary="$RUNS_DIR/$exp/summary.json"
    if summary_complete "$summary" "$exp" "$seed"; then
        echo "[skip] $exp"
        return
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/recover_cifar_summary.py" \
        --run_dir "$RUNS_DIR/$exp" --exp_name "$exp" --seed "$seed"
    if summary_complete "$summary" "$exp" "$seed"; then
        echo "[recovered] $exp"
        return
    fi
    echo "[run] $exp"
    run_one "$exp" "$seed" "$epochs" "$lr" "$patience" "$factor"
    summary_complete "$summary" "$exp" "$seed" || {
        echo "ERROR: missing/incomplete summary: $summary" >&2
        exit 1
    }
}

echo "CIFAR10 NON-AUG HIDDENPROBE: seed=9, 24 configs x 15 epochs"
echo "lr = probe_lr throughout sweep and final runs"
for lr in "${LRS[@]}"; do
    for patience in "${PATIENCES[@]}"; do
        for factor in "${FACTORS[@]}"; do
            exp="${PREFIX}_SWEEP15_s9_lr${lr}_pat${patience}_fac${factor}"
            run_if_needed "$exp" "$SWEEP_SEED" "$SWEEP_EPOCHS" \
                "$lr" "$patience" "$factor"
        done
    done
done

