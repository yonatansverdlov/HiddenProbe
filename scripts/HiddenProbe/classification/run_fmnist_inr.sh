#!/usr/bin/env bash
set -euo pipefail

# Fashion-MNIST INR HiddenProbe: 24-config plateau sweep followed by a five-seed run.
# Run from anywhere: bash scripts/HiddenProbe/classification/run_fmnist_inr.sh
# Completed configurations are skipped so interrupted sweeps can be resumed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${SKIP_DATA_SETUP:-0}" != "1" ]]; then
    bash scripts/setup_data/classification_fmnist.sh
fi

# Keep the environment used by the current merged classification scripts.
if [[ "${SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-probegen}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_PY="${MAIN_PY:-main.py}"
PREFIX="${PREFIX:-fmnist_hidden_mlp2}"
RUN_ROOT="${RUN_ROOT:-experiments/classification/fmnist/runs}"
SWEEP_DIR="$RUN_ROOT/sweeps/$PREFIX"
RANKED_CSV="$SWEEP_DIR/ranked_24.csv"
BEST_ENV="$SWEEP_DIR/best.env"
mkdir -p "$SWEEP_DIR"

SWEEP_EPOCHS=20
FINAL_EPOCHS=30
LRS=(3e-4 5e-4 7e-4)
PATIENCES=(3 5)
FACTORS=(0.2 0.3 0.5 0.7)

# Explicit architecture/training values: do not depend on trainer defaults.
BATCH_SIZE=64
N_PROBES=128
D_HID=256
MIXER_LAYERS=6
GEN_TYPE=linear_2_no_acts
GEN_LATENT_Z=32
GENERATOR_WIDTH=16
PER_PROBE_MLP=mlp2
PER_PROBE_MLP_WIDTH=256
PER_PROBE_OUT_DIM=4
PER_PROBE_INIT=standard
R_PER_HIDDEN=2
RANK=8
SCHEDULER=plateau
PLATEAU_MONITOR=val_acc
PLATEAU_MIN_LR=1e-6
WEIGHT_DECAY=0.0
EVAL_EVERY=500
N_WORKERS=0
DEVICE="${DEVICE:-cuda}"

run_common_args() {
    local exp_name="$1" seed="$2" num_seeds="$3" epochs="$4"
    local lr="$5" patience="$6" factor="$7"

    "$PYTHON_BIN" "$MAIN_PY" \
        --method hiddenprobe \
        --task classification \
        --dataset fmnist \
        --exp_name="$exp_name" \
        --seed="$seed" \
        --num_seeds="$num_seeds" \
        --epochs="$epochs" \
        --batch_size="$BATCH_SIZE" \
        --n_probes="$N_PROBES" \
        --d_hid="$D_HID" \
        --mixer_n_layers="$MIXER_LAYERS" \
        --gen_type="$GEN_TYPE" \
        --gen_latent_z="$GEN_LATENT_Z" \
        --generator_width="$GENERATOR_WIDTH" \
        --per_probe_mlp="$PER_PROBE_MLP" \
        --per_probe_mlp_width="$PER_PROBE_MLP_WIDTH" \
        --per_probe_out_dim="$PER_PROBE_OUT_DIM" \
        --per_probe_init="$PER_PROBE_INIT" \
        --r_per_hidden="$R_PER_HIDDEN" \
        --rank="$RANK" \
        --scheduler="$SCHEDULER" \
        --plateau_monitor="$PLATEAU_MONITOR" \
        --plateau_patience="$patience" \
        --plateau_factor="$factor" \
        --plateau_min_lr="$PLATEAU_MIN_LR" \
        --lr="$lr" \
        --probe_lr="$lr" \
        --weight_decay="$WEIGHT_DECAY" \
        --eval_every="$EVAL_EVERY" \
        --n_workers="$N_WORKERS" \
        --device="$DEVICE"
}

# Only skip a run if a complete, parseable CSV exists for the intended seeds.
summary_complete() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import csv
import math
import sys
from pathlib import Path

path, expected = Path(sys.argv[1]), int(sys.argv[2])
required = {
    "seed", "best_epoch", "best_global_step", "best_val_acc",
    "best_test_acc", "best_val_loss", "best_test_loss",
}
try:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError("missing columns")
        rows = list(reader)
    if len(rows) != expected:
        raise ValueError("wrong number of rows")
    if sorted(int(r["seed"]) for r in rows) != list(range(expected)):
        raise ValueError("unexpected seeds")
    for r in rows:
        int(float(r["best_epoch"]))
        int(float(r["best_global_step"]))
        for key in ("best_val_acc", "best_test_acc", "best_val_loss", "best_test_loss"):
            if not math.isfinite(float(r[key])):
                raise ValueError(f"nonfinite {key}")
except (OSError, ValueError, TypeError, OverflowError):
    sys.exit(1)
PY
}

echo "============================================================"
echo "Fashion-MNIST INR HIDDEN SWEEP — merged backend"
echo "24 configs: 3 LR x 2 patience x 4 factor"
echo "Short runs: seed=0, epochs=$SWEEP_EPOCHS, eval_every=$EVAL_EVERY"
echo "Final: best validation config, seeds 0..4, epochs=$FINAL_EPOCHS"
echo "Results: $SWEEP_DIR"
echo "============================================================"

for lr in "${LRS[@]}"; do
    for patience in "${PATIENCES[@]}"; do
        for factor in "${FACTORS[@]}"; do
            exp_name="${PREFIX}_SWEEP20_plateau_lr${lr}_pat${patience}_fac${factor}"
            summary="$RUN_ROOT/$exp_name/seeds_summary.csv"
            if summary_complete "$summary" 1; then
                echo "[skip] Completed: $exp_name"
            else
                echo "[run] $exp_name"
                run_common_args "$exp_name" 0 1 "$SWEEP_EPOCHS" "$lr" "$patience" "$factor"
                summary_complete "$summary" 1 || {
                    echo "ERROR: Missing or invalid summary: $summary" >&2
                    exit 1
                }
            fi
        done
    done
done

export RUN_ROOT PREFIX RANKED_CSV BEST_ENV
"$PYTHON_BIN" - <<'PY'
import csv
import math
import os
from pathlib import Path

prefix = os.environ["PREFIX"]
run_root = Path(os.environ["RUN_ROOT"])
ranked_csv = Path(os.environ["RANKED_CSV"])
best_env = Path(os.environ["BEST_ENV"]

rows = []
for lr in ("3e-4", "5e-4", "7e-4"):
    for patience in (3, 5):
        for factor in ("0.2", "0.3", "0.5", "0.7"):
            exp_name = f"{prefix}_SWEEP20_plateau_lr{lr}_pat{patience}_fac{factor}"
            summary = run_root / exp_name / "seeds_summary.csv"
            with summary.open(newline="") as f:
                data = list(csv.DictReader(f))
            if len(data) != 1 or int(data[0]["seed"]) != 0:
                raise RuntimeError(f"Expected seed 0 only in {summary}")
            r = data[0]
            val_acc = float(r["best_val_acc"])
            if not math.isfinite(val_acc):
                raise ValueError(f"Nonfinite validation accuracy in {summary}")
            rows.append({
                "exp_name": exp_name,
                "lr": lr,
                "probe_lr": lr,
                "patience": patience,
                "factor": factor,
                "best_epoch": int(float(r["best_epoch"])),
                "best_global_step": int(float(r["best_global_step"])),
                "best_val_acc": val_acc,
                "best_test_acc": float(r["best_test_acc"]),
                "best_val_loss": float(r["best_val_loss"]),
                "best_test_loss": float(r["best_test_loss"]),
            })

# Never select hyperparameters using the test set.
rows.sort(key=lambda x: (-x["best_val_acc"], x["best_epoch"], x["best_global_step"]))
with ranked_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

best = rows[0]
with best_env.open("w") as f:
    f.write(f'BEST_LR="{best["lr"]}"\n')
    f.write(f'BEST_PATIENCE="{best["patience"]}"\n')
    f.write(f'BEST_FACTOR="{best["factor"]}"\n')
    f.write(f'BEST_SWEEP_EXP="{best["exp_name"]}"\n')

print("\n" + "=" * 100)
print("TOP 24 CONFIGS — BY VALIDATION ACCURACY")
print("=" * 100)
for i, r in enumerate(rows, 1):
    print(
        f'{i:2d}. val={r["best_val_acc"]:.6f} test={r["best_test_acc"]:.6f} '
        f'epoch={r["best_epoch"]:2d} step={r["best_global_step"]:6d} '
        f'lr={r["lr"]} probe_lr={r["probe_lr"]} '
        f'pat={r["patience"]} fac={r["factor"]}'
    )
print("\nSELECTED BY VALIDATION ACCURACY ONLY")
print(f'lr = probe_lr:    {best["lr"]}')
print(f'patience:         {best["patience"]}')
print(f'factor:           {best["factor"]}')
print(f'best_val_acc:     {best["best_val_acc"]:.6f}')
print(f'best_test_acc:    {best["best_test_acc"]:.6f} [report only]')
print(f'ranked CSV:       {ranked_csv}')
PY

# Values in best.env are taken only from fixed sweep configuration arrays.
# shellcheck disable=SC1090
source "$BEST_ENV"
FINAL_EXP="${PREFIX}_FINAL_5SEEDS_30E_plateau_lr${BEST_LR}_pat${BEST_PATIENCE}_fac${BEST_FACTOR}"
FINAL_SUMMARY="$RUN_ROOT/$FINAL_EXP/seeds_summary.csv"

echo
echo "============================================================"
echo "FINAL FIVE-SEED RUN — $FINAL_EXP"
echo "Seeds: 0,1,2,3,4; epochs=$FINAL_EPOCHS"
echo "lr=probe_lr=$BEST_LR; patience=$BEST_PATIENCE; factor=$BEST_FACTOR"
echo "============================================================"

if summary_complete "$FINAL_SUMMARY" 5; then
    echo "[skip] Final five-seed run already complete."
else
    run_common_args "$FINAL_EXP" 0 5 "$FINAL_EPOCHS" "$BEST_LR" "$BEST_PATIENCE" "$BEST_FACTOR"
    summary_complete "$FINAL_SUMMARY" 5 || {
        echo "ERROR: Missing or invalid final summary: $FINAL_SUMMARY" >&2
        exit 1
    }
fi

export FINAL_SUMMARY
"$PYTHON_BIN" - <<'PY'
import csv
import os
from statistics import mean, stdev

with open(os.environ["FINAL_SUMMARY"], newline="") as f:
    rows = list(csv.DictReader(f))
seeds = sorted(int(r["seed"]) for r in rows)
assert seeds == [0, 1, 2, 3, 4], seeds
values = [float(r["best_test_acc"]) for r in rows]
print("\nFashion-MNIST / HiddenProbe — final five-seed test accuracy")
print(f"Mean: {mean(values):.4f}")
print(f"Std:  {stdev(values):.4f}")
print(f'Summary: {os.environ["FINAL_SUMMARY"]}')
PY
