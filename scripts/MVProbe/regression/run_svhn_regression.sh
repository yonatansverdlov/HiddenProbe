#!/usr/bin/env bash
set -euo pipefail

# MVProbe on the existing SVHN Small-CNN Zoo.
#
# Stage 1 (seed 0, validation only):
#   4 weight tensors x 2 n_probes x 2 proj_dim x 2 learning rates = 32 configs
#   30 epochs each; TEST IS NEVER LOADED.
#
# Stage 2:
#   select highest best_val_tau, then run that exact configuration on seeds 0..4.
#
# The released MVProbe four-view encoder is preserved; only the task head/data
# adapter are changed from classification to raw-accuracy regression.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

bash scripts/setup_data/regression_svhn.sh

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
DATA_DIR="${SVHN_DIR:-$DATA_ROOT/regression/svhn}"
SPLIT="${SVHN_SPLIT:-$DATA_DIR/split.csv}"
PYTHON="${PYTHON:-python}"

[[ -s "$DATA_DIR/weights.npy" ]] || { echo "Missing $DATA_DIR/weights.npy" >&2; exit 2; }
[[ -s "$DATA_DIR/layout.csv" ]] || { echo "Missing $DATA_DIR/layout.csv" >&2; exit 2; }
[[ -s "$DATA_DIR/metrics.csv.gz" ]] || { echo "Missing $DATA_DIR/metrics.csv.gz" >&2; exit 2; }
[[ -s "$SPLIT" ]] || { echo "Missing $SPLIT" >&2; exit 2; }

SWEEP_EPOCHS="${SWEEP_EPOCHS:-30}"
FINAL_EPOCHS="${FINAL_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
REP_DIM="${REP_DIM:-512}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
DEVICE="${DEVICE:-cuda}"

SWEEP_ROOT="${OUT_DIR:-checkpoints}/mvprobe_svhn_sweep_s0"
FINAL_ROOT="${OUT_DIR:-checkpoints}/mvprobe_svhn_best"
mkdir -p "$SWEEP_ROOT" "$FINAL_ROOT"

RUN=0
TOTAL=32
for LAYER in 0 1 2 3; do
  for N_PROBES in 64 128; do
    for PROJ_DIM in 64 128; do
      for LR in 1e-4 3e-4; do
        RUN=$((RUN + 1))
        NAME="mvprobe_svhn_L${LAYER}_Q${N_PROBES}_P${PROJ_DIM}_lr${LR}_s0"
        DIR="$SWEEP_ROOT/$NAME"
        echo "========== SWEEP [$RUN/$TOTAL] layer=$LAYER Q=$N_PROBES proj=$PROJ_DIM lr=$LR =========="
        if [[ -s "$DIR/summary.json" ]]; then
          echo "Completed: $NAME (skipping)"
          continue
        fi
        "$PYTHON" models/mvprobe_svhn_trainer.py           --data_dir "$DATA_DIR"           --split_csv "$SPLIT"           --layer_index "$LAYER"           --n_probes "$N_PROBES"           --proj_dim "$PROJ_DIM"           --rep_dim "$REP_DIM"           --lr "$LR"           --weight_decay "$WEIGHT_DECAY"           --batch_size "$BATCH_SIZE"           --epochs "$SWEEP_EPOCHS"           --seed 0           --device "$DEVICE"           --skip_test_eval           --out_dir "$DIR"
      done
    done
  done
done

"$PYTHON" - "$SWEEP_ROOT" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
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
        "best_val_tau": float(s["best_val_tau"]),
        "best_epoch": int(s["best_epoch"]),
        "params": int(s["params"]),
    })

if len(rows) != 32:
    raise RuntimeError(f"Expected 32 completed sweep configs, found {len(rows)}")

rows.sort(key=lambda r: r["best_val_tau"], reverse=True)
fields = list(rows[0])
with (root / "sweep_results.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

best = rows[0]
(root / "selected_config.json").write_text(json.dumps(best, indent=2) + "\n")
print("\n========== SELECTED BY VALIDATION TAU ==========")
for k, v in best.items():
    print(f"{k}: {v}")
print(f"Ranked results: {root / 'sweep_results.csv'}")
PY

read -r BEST_LAYER BEST_Q BEST_PROJ BEST_REP BEST_LR < <(
  "$PYTHON" - "$SWEEP_ROOT/selected_config.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(s["layer_index"], s["n_probes"], s["proj_dim"], s["rep_dim"], s["lr"])
PY
)

echo
echo "========== FINAL CONFIG =========="
echo "layer=$BEST_LAYER Q=$BEST_Q proj_dim=$BEST_PROJ rep_dim=$BEST_REP lr=$BEST_LR epochs=$FINAL_EPOCHS"
echo "=================================="

SUMMARIES=()
for SEED in 0 1 2 3 4; do
  NAME="mvprobe_svhn_L${BEST_LAYER}_Q${BEST_Q}_P${BEST_PROJ}_lr${BEST_LR}_s${SEED}"
  DIR="$FINAL_ROOT/$NAME"
  SUMMARIES+=("$DIR/summary.json")
  if [[ -s "$DIR/summary.json" ]]; then
    echo "Completed final seed $SEED (skipping)"
    continue
  fi
  "$PYTHON" models/mvprobe_svhn_trainer.py     --data_dir "$DATA_DIR"     --split_csv "$SPLIT"     --layer_index "$BEST_LAYER"     --n_probes "$BEST_Q"     --proj_dim "$BEST_PROJ"     --rep_dim "$BEST_REP"     --lr "$BEST_LR"     --weight_decay "$WEIGHT_DECAY"     --batch_size "$BATCH_SIZE"     --epochs "$FINAL_EPOCHS"     --seed "$SEED"     --device "$DEVICE"     --out_dir "$DIR"
done

"$PYTHON" - "$SWEEP_ROOT/selected_config.json" "${SUMMARIES[@]}" <<'PY'
import json
import sys
from pathlib import Path

from models.logging_utils import print_final_summary

selected = json.load(open(sys.argv[1]))
values = []
for item in sys.argv[2:]:
    summary = json.load(open(item))
    values.append(float(summary["final_test_tau"]))

print("\nSelected MVProbe configuration:")
print(
    f"layer={selected['layer_index']} ({selected['layer_name']}), "
    f"X={selected['matrix_shape']}, Q={selected['n_probes']}, "
    f"proj_dim={selected['proj_dim']}, rep_dim={selected['rep_dim']}, "
    f"lr={selected['lr']}, sweep_val_tau={selected['best_val_tau']:.4f}"
)
print_final_summary(
    method="MVProbe",
    task="regression",
    dataset="SVHN",
    values=values,
)
PY
