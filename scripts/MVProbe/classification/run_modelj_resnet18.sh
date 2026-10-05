#!/usr/bin/env bash
set -euo pipefail

# MVProbe on the matched Model-J ResNet18 zoo.
#
# IMPORTANT: this is a LAYER SWEEP ONLY.
# All non-layer hyperparameters are kept at the configuration that worked on
# Model-J ResNet101 in the released MVProbe experiments:
#   n_probes=128, proj_dim=128, rep_dim=512
#   lr=3e-4, weight_decay=1e-5, batch_size=128
#   epochs=3000, validation every 25 epochs
#
# We select the ResNet18 layer using validation only.
# Test is never evaluated during this sweep.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${MODELJ_RESNET_ROOT:-$REPO_ROOT/data/classification/modelj_cifar100_resnet}"
ARCH="resnet18"

# ResNet101 working configuration.
N_PROBES="${N_PROBES:-128}"
PROJ_DIM="${PROJ_DIM:-128}"
REP_DIM="${REP_DIM:-512}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-3000}"
EVAL_EVERY="${EVAL_EVERY:-25}"
NUM_WORKERS="${NUM_WORKERS:-4}"

SWEEP_SEED="${SWEEP_SEED:-1}"
SWEEP_ROOT="${OUT_DIR:-checkpoints}/mvprobe_modelj_resnet18_layer_sweep_s${SWEEP_SEED}"
mkdir -p "$SWEEP_ROOT"

[[ -d "$DATA_ROOT/$ARCH/train" ]] || { echo "Missing $DATA_ROOT/$ARCH/train" >&2; exit 2; }

read -r N_LAYERS < <(
  "$PYTHON" - "$DATA_ROOT" "$ARCH" <<'PY'
import sys
from models.modelj_local_dataset import list_model_files
from models.modelj_probe_trainer import modelj_probe_layers

root, arch = sys.argv[1:]
expected = {"train": 701, "val": 100, "test": 201}
for split, n in expected.items():
    got = len(list_model_files(root, arch, split))
    if got != n:
        raise RuntimeError(f"{arch}/{split}: expected {n} models, found {got}")

layers = modelj_probe_layers(root, arch)
print(len(layers))
PY
)

echo "MVProbe Model-J $ARCH: $N_LAYERS probe-compatible layers"
echo "Fixed ResNet101 config: Q=$N_PROBES proj=$PROJ_DIM rep=$REP_DIM lr=$LR bs=$BATCH_SIZE wd=$WEIGHT_DECAY epochs=$EPOCHS"

is_complete_summary() {
  local summary="$1"
  local required_key="$2"
  [[ -s "$summary" ]] || return 1
  "$PYTHON" - "$summary" "$required_key" <<'PY' >/dev/null 2>&1
import json, sys
s = json.load(open(sys.argv[1]))
required = {"best_epoch", "params", sys.argv[2]}
if not required.issubset(s):
    raise SystemExit(1)
PY
}

for ((LAYER=0; LAYER<N_LAYERS; LAYER++)); do
  RUN=$((LAYER + 1))
  NAME="mvprobe_modelj_resnet18_L${LAYER}_s${SWEEP_SEED}"
  DIR="$SWEEP_ROOT/$NAME"

  echo
  echo "================================================================================"
  echo "LAYER SWEEP $RUN/$N_LAYERS | layer=$LAYER | seed=$SWEEP_SEED"
  echo "================================================================================"

  if is_complete_summary "$DIR/summary.json" "best_val_acc"; then
    echo "Already completed: $NAME"
    continue
  fi

  "$PYTHON" models/modelj_probe_trainer.py \
    --model_variant mvprobe \
    --root "$DATA_ROOT" \
    --architecture "$ARCH" \
    --layer_index "$LAYER" \
    --n_probes "$N_PROBES" \
    --proj_dim "$PROJ_DIM" \
    --rep_dim "$REP_DIM" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --eval_every "$EVAL_EVERY" \
    --seed "$SWEEP_SEED" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --skip_test_eval \
    --out_dir "$DIR"
done

"$PYTHON" - "$SWEEP_ROOT" "$N_LAYERS" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
rows = []

for p in root.glob("*/summary.json"):
    s = json.loads(p.read_text())
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
        "best_val_acc": float(s["best_val_acc"]),
        "best_val_loss": float(s["best_val_loss"]),
        "best_epoch": int(s["best_epoch"]),
        "params": int(s["params"]),
    })

if len(rows) != expected:
    raise RuntimeError(f"Expected {expected} layer summaries, found {len(rows)}")

rows.sort(key=lambda r: r["best_val_acc"], reverse=True)

with (root / "layer_results.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

(root / "selected_layer.json").write_text(json.dumps(rows[0], indent=2) + "\n")

print("\n========== SELECTED LAYER BY VALIDATION ACCURACY ==========")
for k, v in rows[0].items():
    print(f"{k}: {v}")
print(f"Ranked results: {root / 'layer_results.csv'}")
PY

echo
echo "Layer sweep complete."
echo "Ranked validation results: $SWEEP_ROOT/layer_results.csv"
echo "Selected layer: $SWEEP_ROOT/selected_layer.json"
