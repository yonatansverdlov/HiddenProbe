#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--exp_name", required=True)
p.add_argument("--seed", required=True, type=int)
args = p.parse_args()

run_dir = Path(args.run_dir)
summary = run_dir / "summary.json"
log = run_dir / "log.csv"
best = run_dir / "best.pt"

if summary.exists():
    raise SystemExit(0)

if not log.exists() or not best.exists():
    raise SystemExit(0)

with log.open(newline="") as f:
    rows = list(csv.DictReader(f))

final_rows = [r for r in rows if r.get("exp_name") == "FINAL"]
if not final_rows:
    raise SystemExit(0)

r = final_rows[-1]
val = float(r["val_acc"])
test = float(r["test_acc"])

with summary.open("w") as f:
    json.dump(
        {
            "exp": args.exp_name,
            "seed": args.seed,
            "best_val_acc": val,
            "best_test_acc": test,
            "final_val_acc": val,
            "recovered_from_log": True,
        },
        f,
        indent=2,
    )

print(f"[recover] restored {summary} from completed log.csv")
