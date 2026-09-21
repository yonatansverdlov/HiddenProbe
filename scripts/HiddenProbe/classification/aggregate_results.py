#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", default="HiddenProbe")
parser.add_argument("summaries", nargs="+")
args = parser.parse_args()

values = []
for item in args.summaries:
    path = Path(item)
    with path.open() as f:
        summary = json.load(f)
    values.append(float(summary["best_test_acc"]))

mean = statistics.fmean(values)
std = statistics.stdev(values) if len(values) > 1 else 0.0

print(f"{args.dataset} — {args.model}")
print(f"Test accuracy: {mean:.4f} ± {std:.4f}")
print("Seeds: " + ", ".join(f"{v:.4f}" for v in values))
