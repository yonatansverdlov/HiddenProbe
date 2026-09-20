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

taus = []
for p in args.summaries:
    path = Path(p)
    with path.open() as f:
        summary = json.load(f)
    if "final_test_tau" not in summary:
        raise RuntimeError(f"{path} does not contain final_test_tau")
    taus.append(float(summary["final_test_tau"]))

mean = statistics.fmean(taus)
std = statistics.pstdev(taus)

print(f"Dataset: {args.dataset}")
print(f"Model: {args.model}")
print(f"Mean: {mean:.4f}")
print(f"Std:  {std:.4f}")
