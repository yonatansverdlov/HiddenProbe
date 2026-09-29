#!/usr/bin/env bash
set -euo pipefail

# Run from any directory: bash scripts/HiddenProbe/regression/sweep_svhn_probe_lr.sh [N_PROBES]
# Matches run_svhn_regression.sh except swept probe_lr, plateau_factor, plateau_patience.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MAIN_PY="${MAIN_PY:-main.py}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
ZOO_DIR="${ZOO_DIR:-$DATA_ROOT/regression/svhn}"
SPLIT="$ZOO_DIR/split.csv"
N_PROBES="${1:-128}"
SEED=0
SWEEP_ROOT="${OUT_DIR:-checkpoints}/svhn_probe_lr_sweep_Q${N_PROBES}_s${SEED}"

bash scripts/setup_data/regression_svhn.sh
[[ -s "$SPLIT" ]] || { echo "Missing SVHN split: $SPLIT" >&2; exit 2; }
mkdir -p "$SWEEP_ROOT"

RUN=0
TOTAL=24
for PROBE_LR in 1e-4 3e-4 6e-4; do
  for FACTOR in 0.2 0.3 0.5 0.9; do
    for PATIENCE in 4 5; do
      RUN=$((RUN + 1))
      EXP_NAME="hiddenprobe_svhn_Q${N_PROBES}_plr${PROBE_LR}_f${FACTOR}_p${PATIENCE}_s${SEED}"
      EXP_DIR="$SWEEP_ROOT/$EXP_NAME"

      echo "========== [$RUN/$TOTAL] probe_lr=$PROBE_LR factor=$FACTOR patience=$PATIENCE seed=$SEED =========="
      if [[ -s "$EXP_DIR/summary.json" ]]; then
        echo "Completed: $EXP_NAME (skipping)"
        continue
      fi

      python "$MAIN_PY" \
        --method hiddenprobe --task regression --dataset svhn \
        --zoo svhn_gs --gen_type deep_linear_6 --models_c_in 1 \
        --zoo_data_dir "$ZOO_DIR" --zoo_split "$SPLIT" \
        --hidden_mode on --probe_sharing shared --assert_canonical 1 --target_space raw \
        --adapter_preset expressive --interaction_rank 96 --mixer_hidden 384 --hidden_dim 0 \
        --n_probes "$N_PROBES" \
        --lr 3e-4 --probe_lr "$PROBE_LR" --hidden_lr 0 \
        --batch_size 32 --rank_loss_w 0.0 --weight_decay 0.0 \
        --scheduler plateau --plateau_factor "$FACTOR" --plateau_patience "$PATIENCE" --plateau_min_lr 3e-5 \
        --epochs 20 --eval_every 1000 --eval_cnn_bs 256 \
        --seed "$SEED" --skip_test_eval 1 --exp_name "$EXP_NAME" --out_dir "$EXP_DIR"
    done
  done
done

# Select the winning configuration using validation tau ONLY.
# Test data are not evaluated during tuning.
python - "$SWEEP_ROOT" <<'PY'
import csv
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
pattern = re.compile(r"hiddenprobe_svhn_Q(?P<n_probes>\d+)_plr(?P<probe_lr>[^_]+)_f(?P<factor>[^_]+)_p(?P<patience>\d+)_s(?P<seed>\d+)")
rows = []
for path in root.glob("*/summary.json"):
    match = pattern.fullmatch(path.parent.name)
    if match is None:
        continue
    result = json.loads(path.read_text())
    if "best_val_tau" not in result:
        raise RuntimeError(f"Incomplete summary: {path}")
    rows.append({
        **match.groupdict(),
        "best_val_tau": float(result["best_val_tau"]),
        "best_epoch": result.get("best_epoch", ""),
        "best_step": result.get("best_step", ""),
    })

if len(rows) != 24:
    raise RuntimeError(f"Expected 24 completed sweep configurations, found {len(rows)}")
rows.sort(key=lambda x: x["best_val_tau"], reverse=True)

csv_path = root / "sweep_results.csv"
fields = ["n_probes", "probe_lr", "factor", "patience", "seed", "best_val_tau", "best_epoch", "best_step"]
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

best = rows[0]
selected = {
    "lr": 3e-4,
    "probe_lr": best["probe_lr"],
    "plateau_factor": best["factor"],
    "plateau_patience": int(best["patience"]),
    "selected_by": "highest best_val_tau on sweep seed 0",
    "sweep_best_val_tau": best["best_val_tau"],
}
selection_path = root / "selected_config.json"
selection_path.write_text(json.dumps(selected, indent=2) + "\n")
print(f"Completed 24/24 sweep configurations. Ranked CSV: {csv_path}")
print(f"SELECTED: lr=3e-4 probe_lr={best['probe_lr']} "
      f"factor={best['factor']} patience={best['patience']} "
      f"val_tau={best['best_val_tau']:.4f}")
print(f"Selected configuration saved to {selection_path}")
PY

BEST_SETTINGS="$(python - "$SWEEP_ROOT/selected_config.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    config = json.load(f)
print(config["probe_lr"], config["plateau_factor"], config["plateau_patience"])
PY
)"
read -r BEST_PROBE_LR BEST_FACTOR BEST_PATIENCE <<< "$BEST_SETTINGS"

# Full 5-seed evaluation of the single selected configuration.
FINAL_ROOT="${OUT_DIR:-checkpoints}/svhn_probe_lr_best_Q${N_PROBES}"
mkdir -p "$FINAL_ROOT"
SUMMARIES=()
echo "========== BEST CONFIG: lr=3e-4 probe_lr=$BEST_PROBE_LR factor=$BEST_FACTOR patience=$BEST_PATIENCE =========="
for FINAL_SEED in 0 1 2 3 4; do
  EXP_NAME="hiddenprobe_svhn_best_Q${N_PROBES}_s${FINAL_SEED}"
  EXP_DIR="$FINAL_ROOT/$EXP_NAME"
  SUMMARIES+=("$EXP_DIR/summary.json")

  if [[ -s "$EXP_DIR/summary.json" ]]; then
    echo "Completed: $EXP_NAME (skipping)"
    continue
  fi

  echo "========== FINAL SEED $FINAL_SEED / 4 =========="
  python "$MAIN_PY" \
    --method hiddenprobe --task regression --dataset svhn \
    --zoo svhn_gs --gen_type deep_linear_6 --models_c_in 1 \
    --zoo_data_dir "$ZOO_DIR" --zoo_split "$SPLIT" \
    --hidden_mode on --probe_sharing shared --assert_canonical 1 --target_space raw \
    --adapter_preset expressive --interaction_rank 96 --mixer_hidden 384 --hidden_dim 0 \
    --n_probes "$N_PROBES" \
    --lr 3e-4 --probe_lr "$BEST_PROBE_LR" --hidden_lr 0 \
    --batch_size 32 --rank_loss_w 0.0 --weight_decay 0.0 \
    --scheduler plateau --plateau_factor "$BEST_FACTOR" --plateau_patience "$BEST_PATIENCE" --plateau_min_lr 3e-5 \
    --epochs 150 --eval_every 500 --eval_cnn_bs 256 \
    --seed "$FINAL_SEED" --skip_test_eval 0 --exp_name "$EXP_NAME" --out_dir "$EXP_DIR"
done

python - "${SUMMARIES[@]}" <<'PY' | tee "$FINAL_ROOT/final_5seeds.log"
import json
import sys
from pathlib import Path

from models.logging_utils import print_final_summary

values = []
for item in sys.argv[1:]:
    with Path(item).open() as f:
        summary = json.load(f)
    if "final_test_tau" not in summary:
        raise RuntimeError(f"{item} does not contain final_test_tau")
    values.append(float(summary["final_test_tau"]))

print_final_summary(
    method="HiddenProbe",
    task="regression",
    dataset="SVHN",
    values=values,
)
PY
