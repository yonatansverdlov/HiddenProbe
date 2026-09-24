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
        --epochs 150 --eval_every 500 --eval_cnn_bs 256 \
        --seed "$SEED" --exp_name "$EXP_NAME" --out_dir "$EXP_DIR"
    done
  done
done

# Rank configurations by validation tau, not by test performance.
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
        "final_test_tau": result.get("final_test_tau", ""),
        "best_epoch": result.get("best_epoch", ""),
        "best_step": result.get("best_step", ""),
    })

rows.sort(key=lambda x: x["best_val_tau"], reverse=True)
out = root / "sweep_results.csv"
fields = ["n_probes", "probe_lr", "factor", "patience", "seed", "best_val_tau", "final_test_tau", "best_epoch", "best_step"]
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"Completed {len(rows)}/24 configurations. Results: {out}")
for row in rows[:5]:
    print(f"probe_lr={row['probe_lr']} factor={row['factor']} patience={row['patience']} "
          f"val_tau={row['best_val_tau']:.4f}")
PY
