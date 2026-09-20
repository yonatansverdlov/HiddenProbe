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

print(f"Dataset: {args.dataset}")
print(f"Model: {args.model}")
print(f"Mean: {statistics.fmean(values):.4f}")
print(f"Std:  {statistics.pstdev(values):.4f}")
