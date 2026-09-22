#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from models.logging_utils import print_final_summary

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

print_final_summary(
    method=args.model,
    task="classification",
    dataset=args.dataset,
    values=values,
)
