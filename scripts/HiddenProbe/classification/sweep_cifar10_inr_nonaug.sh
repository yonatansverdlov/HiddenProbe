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

export RUNS_DIR PREFIX RANKED_CSV BEST_ENV
"$PYTHON_BIN" - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

root = Path(os.environ["RUNS_DIR"])
prefix = os.environ["PREFIX"]
ranked = Path(os.environ["RANKED_CSV"])
best_env = Path(os.environ["BEST_ENV"])
rows = []

for lr in ("3e-4", "5e-4", "7e-4"):
    for patience in (3, 5):
        for factor in ("0.2", "0.3", "0.5", "0.7"):
            exp = f"{prefix}_SWEEP15_s9_lr{lr}_pat{patience}_fac{factor}"
            folder = root / exp
            with (folder / "summary.json").open() as f:
                summary = json.load(f)
            if int(summary["seed"]) != 9 or summary["exp"] != exp:
                raise ValueError(f"Unexpected summary contents: {exp}")
            # CIFAR stores epoch and global step in log.csv, not summary.json.
            with (folder / "log.csv").open(newline="") as f:
                logs = list(csv.DictReader(f))
            improved = [
                r for r in logs
                if r.get("is_best", "").strip().lower() in ("true", "1")
            ]
            if not improved:
                raise RuntimeError(f"No best-validation log row: {exp}")
            best_log = max(improved, key=lambda r: float(r["val_acc"]))
            val = float(summary["best_val_acc"])
            test = float(summary["best_test_acc"])
            if not (math.isfinite(val) and math.isfinite(test)):
                raise ValueError(f"Nonfinite result: {exp}")
            rows.append({
                "exp_name": exp,
                "seed": 9,
                "lr": lr,
                "probe_lr": lr,
                "patience": patience,
                "factor": factor,
                "best_epoch": int(best_log["epoch"]) + 1,
                "best_global_step": int(best_log["global_step"]),
                "best_val_acc": val,
                "best_test_acc": test,
            })

if len(rows) != 24:
    raise RuntimeError(f"Expected 24 configurations, got {len(rows)}")

# Selection uses validation accuracy only. Test accuracy is for reporting.
rows.sort(key=lambda r: (-r["best_val_acc"], r["best_epoch"], r["best_global_step"]))
with ranked.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

best = rows[0]
with best_env.open("w") as f:
    f.write(f'BEST_LR="{best["lr"]}"\n')
    f.write(f'BEST_PATIENCE="{best["patience"]}"\n')
    f.write(f'BEST_FACTOR="{best["factor"]}"\n')

print("\n" + "=" * 98)
print("TOP 24 CONFIGURATIONS — RANKED BY VALIDATION ACCURACY")
print("=" * 98)
for rank, r in enumerate(rows, 1):
    print(
        f'{rank:2d}. val={r["best_val_acc"]:.6f} '
        f'test={r["best_test_acc"]:.6f} [report only] '
        f'epoch={r["best_epoch"]:2d} step={r["best_global_step"]:6d} '
        f'lr=probe_lr={r["lr"]} patience={r["patience"]} factor={r["factor"]}'
    )
print(
    f'\nSELECTED: lr=probe_lr={best["lr"]}, '
    f'patience={best["patience"]}, factor={best["factor"]}, '
    f'best_val_acc={best["best_val_acc"]:.6f}'
)
print(f"Ranking CSV: {ranked}")
PY

# Only fixed sweep-grid values are written to best.env.
# shellcheck disable=SC1090
source "$BEST_ENV"
echo
echo "FINAL FIVE-SEED RUN: seeds 0..4, 60 epochs"
echo "lr = probe_lr = $BEST_LR, patience=$BEST_PATIENCE, factor=$BEST_FACTOR"

FINAL_SUMMARIES=()
for seed in 0 1 2 3 4; do
    exp="${PREFIX}_FINAL60_lr${BEST_LR}_pat${BEST_PATIENCE}_fac${BEST_FACTOR}_s${seed}"
    run_if_needed "$exp" "$seed" "$FINAL_EPOCHS" \
        "$BEST_LR" "$BEST_PATIENCE" "$BEST_FACTOR"
    FINAL_SUMMARIES+=("$RUNS_DIR/$exp/summary.json")
done

"$PYTHON_BIN" - "$FINAL_CSV" "${FINAL_SUMMARIES[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

out = Path(sys.argv[1])
rows = []
for path in sys.argv[2:]:
    with Path(path).open() as f:
        s = json.load(f)
    rows.append({
        "seed": int(s["seed"]),
        "best_val_acc": float(s["best_val_acc"]),
        "best_test_acc": float(s["best_test_acc"]),
        "exp_name": s["exp"],
    })
rows.sort(key=lambda r: r["seed"])
if [r["seed"] for r in rows] != [0, 1, 2, 3, 4]:
    raise ValueError("Expected exactly five final seeds: 0..4")

with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

values = [r["best_test_acc"] for r in rows]
print("\nCIFAR10 NON-AUG — HIDDENPROBE FINAL TEST ACCURACY")
for r in rows:
    print(f'seed {r["seed"]}: val={r["best_val_acc"]:.6f} test={r["best_test_acc"]:.6f}')
print(f"Mean: {mean(values):.6f}")
print(f"Std:  {stdev(values):.6f} (sample std, ddof=1)")
print(f"Final summary: {out}")
PY
