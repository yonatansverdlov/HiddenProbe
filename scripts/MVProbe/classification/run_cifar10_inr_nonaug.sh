#!/usr/bin/env bash
set -euo pipefail

# MVProbe CIFAR10 non-aug INR classification.
# Released four-view implementation with ReduceLROnPlateau for our sweep protocol.
# Sweep: hidden weights only; final output/RGB layer is excluded.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

bash scripts/setup_data/classification_cifar10.sh

DATA_DIR="${CIFAR10_INR_DIR:-$REPO_ROOT/data/classification/cifar10_inr}"
SPLIT="${CIFAR10_INR_SPLIT:-$DATA_DIR/nfn_cifar_split_noaug.json}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-5}"
PLATEAU_MIN_LR="${PLATEAU_MIN_LR:-1e-6}"
REP_DIM="${REP_DIM:-512}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-60}"
FINAL_EPOCHS="${FINAL_EPOCHS:-150}"

[[ -d "$DATA_DIR" ]] || { echo "Missing $DATA_DIR" >&2; exit 2; }
[[ -s "$SPLIT" ]] || { echo "Missing $SPLIT" >&2; exit 2; }

SWEEP_ROOT="${OUT_DIR:-checkpoints}/mvprobe_cifar10_inr_nonaug_sweep_plateau_s0"
FINAL_ROOT="${OUT_DIR:-checkpoints}/mvprobe_cifar10_inr_nonaug_best_plateau"
mkdir -p "$SWEEP_ROOT" "$FINAL_ROOT"

RUN=0
TOTAL=1152
for LAYER in 0 1 2 3; do
  for N_PROBES in 64 128; do
    for PROJ_DIM in 64 128; do
      for LR in 1e-4 3e-4 5e-4; do
        for BATCH_SIZE in 64 128; do
          for WEIGHT_DECAY in 0 1e-5 1e-4; do
            for PLATEAU_FACTOR in 0.2 0.3 0.5 0.7; do
              RUN=$((RUN + 1))
            NAME="mvprobe_cifar10_inr_nonaug_L${LAYER}_Q${N_PROBES}_P${PROJ_DIM}_lr${LR}_bs${BATCH_SIZE}_wd${WEIGHT_DECAY}_fac${PLATEAU_FACTOR}_s0"
            DIR="$SWEEP_ROOT/$NAME"
            echo "========== SWEEP [$RUN/$TOTAL] layer=$LAYER Q=$N_PROBES proj=$PROJ_DIM lr=$LR bs=$BATCH_SIZE wd=$WEIGHT_DECAY factor=$PLATEAU_FACTOR =========="
            if [[ -s "$DIR/summary.json" ]]; then
              echo "Completed: $NAME (skipping)"
              continue
            fi
            "$PYTHON" models/mvprobe_inr_trainer.py \
              --data_dir "$DATA_DIR" \
              --split_json "$SPLIT" \
              --dataset_name "CIFAR-10 INR non-aug" \
              --n_classes 10 \
              --layer_index "$LAYER" \
              --n_probes "$N_PROBES" \
              --proj_dim "$PROJ_DIM" \
              --rep_dim "$REP_DIM" \
              --lr "$LR" \
              --weight_decay "$WEIGHT_DECAY" \
              --batch_size "$BATCH_SIZE" \
              --epochs "$SWEEP_EPOCHS" \
              --seed 0 \
              --device "$DEVICE" \
              --skip_test_eval \
              --out_dir "$DIR"
            done
          done
        done
      done
    done
  done
done

"$PYTHON" - "$SWEEP_ROOT" "$TOTAL" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
rows = []
for path in root.glob("*/summary.json"):
    s = json.loads(path.read_text())
    rows.append({
        "layer_index": int(s["layer_index"]),
        "layer_name": s["layer_name"],
        "matrix_shape": "x".join(map(str, s["matrix_shape"])),
        "n_probes": int(s["n_probes"]),
        "proj_dim": int(s["proj_dim"]),
        "rep_dim": int(s["rep_dim"]),
        "lr": float(s["lr"]),
        "batch_size": int(s["batch_size"]),
        "weight_decay": float(s["weight_decay"]),
        "plateau_factor": float(s["plateau_factor"]),
        "best_val_acc": float(s["best_val_acc"]),
        "best_epoch": int(s["best_epoch"]),
        "params": int(s["params"]),
    })

if len(rows) != expected:
    raise RuntimeError(f"Expected {expected} completed sweep configs, found {len(rows)}")

rows.sort(key=lambda r: r["best_val_acc"], reverse=True)
fields = list(rows[0])
with (root / "sweep_results.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

best = rows[0]
(root / "selected_config.json").write_text(json.dumps(best, indent=2) + "\n")
print("\n========== SELECTED BY VALIDATION ACCURACY ==========")
for k, v in best.items():
    print(f"{k}: {v}")
print(f"Ranked results: {root / 'sweep_results.csv'}")
PY

read -r BEST_LAYER BEST_Q BEST_PROJ BEST_REP BEST_LR BEST_BS BEST_WD BEST_FACTOR < <(
  "$PYTHON" - "$SWEEP_ROOT/selected_config.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(
    s["layer_index"], s["n_probes"], s["proj_dim"], s["rep_dim"],
    s["lr"], s["batch_size"], s["weight_decay"], s["plateau_factor"]
)
PY
)

echo
echo "========== FINAL CONFIG =========="
echo "layer=$BEST_LAYER Q=$BEST_Q proj_dim=$BEST_PROJ rep_dim=$BEST_REP lr=$BEST_LR bs=$BEST_BS wd=$BEST_WD factor=$BEST_FACTOR epochs=$FINAL_EPOCHS"
echo "=================================="

SUMMARIES=()
for SEED in 0 1 2 3 4; do
  NAME="mvprobe_cifar10_inr_nonaug_L${BEST_LAYER}_Q${BEST_Q}_P${BEST_PROJ}_lr${BEST_LR}_bs${BEST_BS}_wd${BEST_WD}_fac${BEST_FACTOR}_s${SEED}"
  DIR="$FINAL_ROOT/$NAME"
  SUMMARIES+=("$DIR/summary.json")
  if [[ -s "$DIR/summary.json" ]]; then
    echo "Completed final seed $SEED (skipping)"
    continue
  fi
  "$PYTHON" models/mvprobe_inr_trainer.py \
    --data_dir "$DATA_DIR" \
    --split_json "$SPLIT" \
    --dataset_name "CIFAR-10 INR non-aug" \
    --n_classes 10 \
    --layer_index "$BEST_LAYER" \
    --n_probes "$BEST_Q" \
    --proj_dim "$BEST_PROJ" \
    --rep_dim "$BEST_REP" \
    --lr "$BEST_LR" \
    --scheduler plateau \
    --plateau_factor "$BEST_FACTOR" \
    --plateau_patience "$PLATEAU_PATIENCE" \
    --plateau_min_lr "$PLATEAU_MIN_LR" \
    --weight_decay "$BEST_WD" \
    --batch_size "$BEST_BS" \
    --epochs "$FINAL_EPOCHS" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --out_dir "$DIR"
done

"$PYTHON" - "$SWEEP_ROOT/selected_config.json" "${SUMMARIES[@]}" <<'PY'
import json
import sys

from models.logging_utils import print_final_summary

selected = json.load(open(sys.argv[1]))
values = [float(json.load(open(p))["final_test_acc"]) for p in sys.argv[2:]]

print("\nSelected MVProbe configuration:")
print(
    f"layer={selected['layer_index']} ({selected['layer_name']}), "
    f"X={selected['matrix_shape']}, Q={selected['n_probes']}, "
    f"proj_dim={selected['proj_dim']}, rep_dim={selected['rep_dim']}, "
    f"lr={selected['lr']}, batch_size={selected['batch_size']}, "
    f"weight_decay={selected['weight_decay']}, "
    f"plateau_factor={selected['plateau_factor']}, "
    f"sweep_val_acc={selected['best_val_acc']:.4f}"
)
print_final_summary(
    method="MVProbe",
    task="classification",
    dataset="CIFAR-10 INR non-aug",
    values=values,
)
PY
