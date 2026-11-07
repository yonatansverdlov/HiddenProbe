#!/usr/bin/env python3
"""Build a JSONL manifest from completed local Model-J models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/classification/modelj_cifar100_resnet")
    p.add_argument("--architecture", required=True)
    args = p.parse_args()

    arch_root = Path(args.root) / args.architecture
    rows = []
    for path in sorted(arch_root.glob("*" + "/model_idx_*/metadata.json")):
        try:
            row = json.loads(path.read_text())
        except Exception:
            continue
        if row.get("status") != "complete":
            continue
        row["model_dir"] = str(path.parent)
        row["weights_path"] = str(path.parent / "model.safetensors")
        rows.append(row)

    rows.sort(key=lambda x: (x["split"], int(x["model_idx"])))
    manifest = arch_root / "manifest.jsonl"
    arch_root.mkdir(parents=True, exist_ok=True)
    with manifest.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    by_split = {}
    for row in rows:
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
    print(f"Wrote {len(rows)} models to {manifest}")
    print("Counts:", by_split)


if __name__ == "__main__":
    main()
