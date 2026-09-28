#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

bash scripts/setup_data/classification_cifar10.sh

Q="${1:-128}"
NUM_SEEDS=5
RUNS_DIR="checkpoints/cifar10_inr_aug"
SUMMARIES=()

for ((SEED=0; SEED<NUM_SEEDS; SEED++)); do
  EXP_NAME="hiddenprobe_cifar10_inr_aug_Q${Q}_s${SEED}"
  SUMMARY="$RUNS_DIR/$EXP_NAME/summary.json"

  python - "$RUNS_DIR/$EXP_NAME" "$EXP_NAME" "$SEED" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
exp_name = sys.argv[2]
seed = int(sys.argv[3])
summary = run_dir / "summary.json"
log = run_dir / "log.csv"
best = run_dir / "best.pt"

if not summary.exists() and log.exists() and best.exists():
    with log.open(newline="") as f:
        rows = list(csv.DictReader(f))
    final_rows = [r for r in rows if r.get("exp_name") == "FINAL"]
    if final_rows:
        row = final_rows[-1]
        val = float(row["val_acc"])
        test = float(row["test_acc"])
        summary.write_text(json.dumps({
            "exp": exp_name,
            "seed": seed,
            "best_val_acc": val,
            "best_test_acc": test,
            "final_val_acc": val,
            "recovered_from_log": True,
        }, indent=2) + "\n")
        print(f"[recover] restored {summary} from completed log.csv")
PY

  if [[ ! -s "$SUMMARY" ]]; then
    python main.py \
      --method hiddenprobe \
      --task classification \
      --dataset cifar10_aug \
      --gen_type linear_2_no_acts --gen_latent_z 32 --generator_width 16 --n_probes "$Q" --domain_tanh 1 \
      --head set_transformer --d 112 --nenc 3 --nheads 8 \
      --ema_decay 0.999 \
      --lr 4e-4 --probe_lr 4e-3 --batch_size 32 --warmup 300 --dropout 0.1 --head_wd 0.1 \
      --scheduler cosine --plateau_min_lr 1e-5 \
      --epochs 12 --eval_every 500 \
      --seed "$SEED" \
      --runs_dir "$RUNS_DIR" \
      --exp_name "$EXP_NAME"
  fi

  SUMMARIES+=("$SUMMARY")
done

python - "CIFAR-10 Augmented" "${SUMMARIES[@]}" <<'PY'
import json
import sys
from pathlib import Path

from models.logging_utils import print_final_summary

dataset = sys.argv[1]
values = []
for item in sys.argv[2:]:
    with Path(item).open() as f:
        values.append(float(json.load(f)["best_test_acc"]))

print_final_summary(
    method="HiddenProbe",
    task="classification",
    dataset=dataset,
    values=values,
)
PY
