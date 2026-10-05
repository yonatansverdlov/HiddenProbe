#!/usr/bin/env bash
set -euo pipefail

# ProbeX on the matched Model-J ResNet18 zoo.
# Protocol: original ProbeX defaults on Model-J (500 epochs, lr=1e-3),
# validation-only layer sweep on seed 0, then seeds 0..4 on the selected layer.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${MODELJ_RESNET_ROOT:-$REPO_ROOT/data/classification/modelj_cifar100_resnet}"
ARCH="resnet18"
N_PROBES="${N_PROBES:-128}"
PROJ_DIM="${PROJ_DIM:-128}"
REP_DIM="${REP_DIM:-512}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-500}"
EVAL_EVERY="${EVAL_EVERY:-25}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SWEEP_SEED="${SWEEP_SEED:-0}"
FINAL_SEEDS="${FINAL_SEEDS:-0 1 2 3 4}"

SWEEP_ROOT="${OUT_DIR:-checkpoints}/probex_modelj_resnet18_layer_sweep_s${SWEEP_SEED}"
FINAL_ROOT="${OUT_DIR:-checkpoints}/probex_modelj_resnet18_best_layer"
mkdir -p "$SWEEP_ROOT" "$FINAL_ROOT"

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

echo "ProbeX Model-J $ARCH: $N_LAYERS probe-compatible layers"

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
  NAME="probex_modelj_resnet18_L${LAYER}_s${SWEEP_SEED}"
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
    --model_variant probex \
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
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
rows = []
for p in root.glob("*/summary.json"):
    s = json.loads(p.read_text())
    rows.append({k: s[k] for k in [
        "layer_index", "layer_name", "matrix_shape", "n_probes", "proj_dim",
        "rep_dim", "lr", "batch_size", "weight_decay", "best_val_acc",
        "best_val_loss", "best_epoch", "params"
    ]})
if len(rows) != expected:
    raise RuntimeError(f"Expected {expected} layer summaries, found {len(rows)}")
rows.sort(key=lambda r: r["best_val_acc"], reverse=True)
with (root / "layer_results.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
(root / "selected_layer.json").write_text(json.dumps(rows[0], indent=2) + "\n")
print("\n========== SELECTED LAYER BY VALIDATION ACCURACY ==========")
for k, v in rows[0].items(): print(f"{k}: {v}")
PY

BEST_LAYER="$($PYTHON - "$SWEEP_ROOT/selected_layer.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["layer_index"])
PY
)"

echo
echo "========== FINAL ProbeX | ResNet18 | layer=$BEST_LAYER | seeds=$FINAL_SEEDS =========="
SUMMARIES=()
for SEED in $FINAL_SEEDS; do
  NAME="probex_modelj_resnet18_L${BEST_LAYER}_s${SEED}"
  DIR="$FINAL_ROOT/$NAME"
  SUMMARIES+=("$DIR/summary.json")
  if is_complete_summary "$DIR/summary.json" "final_test_acc"; then
    echo "Seed $SEED already completed: $NAME"
    continue
  fi
  "$PYTHON" models/modelj_probe_trainer.py \
    --model_variant probex \
    --root "$DATA_ROOT" \
    --architecture "$ARCH" \
    --layer_index "$BEST_LAYER" \
    --n_probes "$N_PROBES" \
    --proj_dim "$PROJ_DIM" \
    --rep_dim "$REP_DIM" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --eval_every "$EVAL_EVERY" \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --out_dir "$DIR"
done

"$PYTHON" - "$SWEEP_ROOT/selected_layer.json" "${SUMMARIES[@]}" <<'PY'
import json, sys
from models.logging_utils import print_final_summary
selected = json.load(open(sys.argv[1]))
values = [float(json.load(open(p))["final_test_acc"]) for p in sys.argv[2:]]
print(f"Selected layer: {selected['layer_index']} {selected['layer_name']} X={selected['matrix_shape']}")
print_final_summary(method="ProbeX", task="classification", dataset="Model-J CIFAR100 / ResNet18", values=values)
PY
