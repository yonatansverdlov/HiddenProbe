#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

bash scripts/setup_data/classification_fmnist.sh

MAIN_PY="${MAIN_PY:-main.py}"

# Fashion-MNIST INR HiddenProbe sweep, ported from inr_classification_branch.
# 24 Plateau configs:
#   lr       = {3e-4, 5e-4, 7e-4}
#   patience = {3, 5}
#   factor   = {0.2, 0.3, 0.5, 0.7}
# Short sweep: seed 0, 20 epochs.
# Final: best validation config, seeds 0..4, 30 epochs.

PREFIX="fmnist_hidden_mlp2"
RUN_ROOT="experiments/classification/fmnist/runs"

SWEEP_EPOCHS=20
FINAL_EPOCHS=30

LRS=(3e-4 5e-4 7e-4)
PATIENCES=(3 5)
FACTORS=(0.2 0.3 0.5 0.7)

# Fixed architecture / protocol. Keep these explicit: do not rely on trainer defaults.
BATCH_SIZE=64
N_PROBES=128
D_HID=256
MIXER_LAYERS=6

GEN_TYPE="linear_2_no_acts"
GEN_LATENT_Z=32
GENERATOR_WIDTH=16

PER_PROBE_MLP="mlp2"
PER_PROBE_MLP_WIDTH=256
PER_PROBE_OUT_DIM=4
PER_PROBE_INIT="standard"

R_PER_HIDDEN=2
RANK=8

SCHEDULER="plateau"
PLATEAU_MONITOR="val_acc"
PLATEAU_MIN_LR=1e-6

WEIGHT_DECAY=0.0
EVAL_EVERY=500
N_WORKERS=0
DEVICE="cuda"

run_common_args() {
    local exp_name="$1"
    local seed="$2"
    local num_seeds="$3"
    local epochs="$4"
    local lr="$5"
    local patience="$6"
    local factor="$7"

    python "${MAIN_PY}" \
        --method hiddenprobe \
        --task classification \
        --dataset fmnist \
        --exp_name="${exp_name}" \
        --seed="${seed}" \
        --num_seeds="${num_seeds}" \
        --epochs="${epochs}" \
        --batch_size="${BATCH_SIZE}" \
        --n_probes="${N_PROBES}" \
        --d_hid="${D_HID}" \
        --mixer_n_layers="${MIXER_LAYERS}" \
        --gen_type="${GEN_TYPE}" \
        --gen_latent_z="${GEN_LATENT_Z}" \
        --generator_width="${GENERATOR_WIDTH}" \
        --per_probe_mlp="${PER_PROBE_MLP}" \
        --per_probe_mlp_width="${PER_PROBE_MLP_WIDTH}" \
        --per_probe_out_dim="${PER_PROBE_OUT_DIM}" \
        --per_probe_init="${PER_PROBE_INIT}" \
        --r_per_hidden="${R_PER_HIDDEN}" \
        --rank="${RANK}" \
        --scheduler="${SCHEDULER}" \
        --plateau_monitor="${PLATEAU_MONITOR}" \
        --plateau_patience="${patience}" \
        --plateau_factor="${factor}" \
        --plateau_min_lr="${PLATEAU_MIN_LR}" \
        --lr="${lr}" \
        --probe_lr="${lr}" \
        --weight_decay="${WEIGHT_DECAY}" \
        --eval_every="${EVAL_EVERY}" \
        --n_workers="${N_WORKERS}" \
        --device="${DEVICE}"
}

echo "============================================================"
echo "Fashion-MNIST INR HIDDEN SWEEP"
echo "24 configs = 3 LR x 2 patience x 4 factor"
echo "probe_lr is fixed equal to lr in every config"
echo "Short runs: seed=0, epochs=${SWEEP_EPOCHS}, eval_every=${EVAL_EVERY}"
echo "Final: best validation config -> 5 seeds x ${FINAL_EPOCHS} epochs"
echo "============================================================"

for lr in "${LRS[@]}"; do
    for patience in "${PATIENCES[@]}"; do
        for factor in "${FACTORS[@]}"; do
            exp_name="${PREFIX}_SWEEP20_plateau_lr${lr}_pat${patience}_fac${factor}"
            echo
            echo "Running: ${exp_name}"
            run_common_args "${exp_name}" 0 1 "${SWEEP_EPOCHS}" "${lr}" "${patience}" "${factor}"
        done
    done
done

RANKED_CSV="/tmp/${PREFIX}_ranked_24.csv"
BEST_ENV="/tmp/${PREFIX}_best.env"
export RUN_ROOT PREFIX RANKED_CSV BEST_ENV

python - <<'PY'
import csv
import os
from pathlib import Path

prefix = os.environ["PREFIX"]
run_root = Path(os.environ["RUN_ROOT"])
ranked_csv = Path(os.environ["RANKED_CSV"])
best_env = Path(os.environ["BEST_ENV"])

lrs = ["3e-4", "5e-4", "7e-4"]
patiences = [3, 5]
factors = ["0.2", "0.3", "0.5", "0.7"]

rows = []
for lr in lrs:
    for patience in patiences:
        for factor in factors:
            exp_name = f"{prefix}_SWEEP20_plateau_lr{lr}_pat{patience}_fac{factor}"
            summary = run_root / exp_name / "seeds_summary.csv"
            if not summary.exists():
                raise FileNotFoundError(f"Missing summary CSV: {summary}")
            with summary.open(newline="") as f:
                data = list(csv.DictReader(f))
            if len(data) != 1:
                raise RuntimeError(f"Expected one seed in {summary}, got {len(data)}")
            r = data[0]
            rows.append({
                "exp_name": exp_name,
                "lr": lr,
                "probe_lr": lr,
                "patience": patience,
                "factor": factor,
                "best_epoch": int(float(r["best_epoch"])),
                "best_global_step": int(float(r["best_global_step"])),
                "best_val_acc": float(r["best_val_acc"]),
                "best_test_acc": float(r["best_test_acc"]),
                "best_val_loss": float(r["best_val_loss"]),
                "best_test_loss": float(r["best_test_loss"]),
            })

rows.sort(key=lambda x: (-x["best_val_acc"], x["best_epoch"], x["best_global_step"]))

with ranked_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

best = rows[0]
with best_env.open("w") as f:
    f.write(f'BEST_LR="{best["lr"]}"\n')
    f.write(f'BEST_PATIENCE="{best["patience"]}"\n')
    f.write(f'BEST_FACTOR="{best["factor"]}"\n')
    f.write(f'BEST_VAL_ACC="{best["best_val_acc"]:.10f}"\n')
    f.write(f'BEST_TEST_ACC="{best["best_test_acc"]:.10f}"\n')
    f.write(f'BEST_SWEEP_EXP="{best["exp_name"]}"\n')

print()
print("=" * 100)
print("TOP 24 CONFIGS — SORTED BY VALIDATION ACCURACY")
print("=" * 100)
for i, r in enumerate(rows, 1):
    print(
        f'{i:2d}. val={r["best_val_acc"]:.6f} test={r["best_test_acc"]:.6f} '
        f'epoch={r["best_epoch"]:2d} step={r["best_global_step"]:6d} '
        f'lr={r["lr"]} probe_lr={r["probe_lr"]} '
        f'pat={r["patience"]} fac={r["factor"]}'
    )

print()
print("WINNER — VALIDATION ACCURACY ONLY")
print(f'lr = probe_lr:       {best["lr"]}')
print(f'plateau_patience:    {best["patience"]}')
print(f'plateau_factor:      {best["factor"]}')
print(f'best_val_acc:        {best["best_val_acc"]:.6f}')
print(f'best_test_acc:       {best["best_test_acc"]:.6f} [REPORT ONLY]')
print(f'ranked csv:          {ranked_csv}')
PY

# shellcheck disable=SC1090
source "${BEST_ENV}"

FINAL_EXP="${PREFIX}_FINAL_5SEEDS_30E_plateau_lr${BEST_LR}_pat${BEST_PATIENCE}_fac${BEST_FACTOR}"

echo
echo "============================================================"
echo "FINAL 5-SEED RUN"
echo "dataset:             fmnist"
echo "seeds:               0,1,2,3,4"
echo "epochs:              ${FINAL_EPOCHS}"
echo "lr = probe_lr:       ${BEST_LR}"
echo "plateau_patience:    ${BEST_PATIENCE}"
echo "plateau_factor:      ${BEST_FACTOR}"
echo "eval_every:          ${EVAL_EVERY}"
echo "============================================================"

run_common_args "${FINAL_EXP}" 0 5 "${FINAL_EPOCHS}" "${BEST_LR}" "${BEST_PATIENCE}" "${BEST_FACTOR}"

FINAL_SUMMARY="${RUN_ROOT}/${FINAL_EXP}/seeds_summary.csv"
export FINAL_SUMMARY

python - <<'PY'
import os
import pandas as pd

df = pd.read_csv(os.environ["FINAL_SUMMARY"])
val = df["best_val_acc"].astype(float)
test = df["best_test_acc"].astype(float)

print()
print(f"Dataset: Fashion-MNIST")
print("Model: HiddenProbe")
print(f"Mean: {test.mean():.4f}")
print(f"Std:  {test.std(ddof=1):.4f}")
PY
