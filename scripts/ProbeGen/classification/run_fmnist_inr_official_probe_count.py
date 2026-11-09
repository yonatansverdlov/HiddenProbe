#!/usr/bin/env python3
"""Run the UNMODIFIED official Kahana et al. ProbeGen FMNIST-INR implementation.

Unlike the merged HiddenProbe trainer, this launches the upstream main.py and
ProbeGen.py at a pinned Git SHA. The sole experimental variables are Q and seed.
The official code controls architecture, optimizer, CosineAnnealingLR,
evaluation cadence and checkpoints. Data are symlinked, never copied.

Usage:
  python scripts/ProbeGen/classification/run_fmnist_inr_official_probe_count.py --dry-run
  python scripts/ProbeGen/classification/run_fmnist_inr_official_probe_count.py
  python scripts/ProbeGen/classification/run_fmnist_inr_official_probe_count.py --probes 128 --seeds 1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_URL = "https://github.com/jonkahana/ProbeGen.git"
# Pinned GitHub main, verified 2026-11-09. Do not silently follow upstream changes.
OFFICIAL_SHA = "e3b8e1485b01dcbccb0c508896d9c2d101a34fa3"


def execute(cmd: list[str], *, cwd: Path, log: Path | None = None) -> None:
    print("+", shlex.join(cmd), flush=True)
    if log is None:
        subprocess.run(cmd, cwd=cwd, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    # Stream the unmodified upstream stdout/stderr to BOTH the terminal and log.
    # Do not hide progress or change the official model/training implementation.
    with log.open("w", buffering=1) as out:
        with subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                out.write(line)
            returncode = process.wait()
    if returncode:
        raise RuntimeError(f"Upstream training exited with {returncode}; see {log}")


def prepare_upstream(directory: Path) -> None:
    if not (directory / ".git").is_dir():
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"{directory} is nonempty but is not a git checkout")
        directory.parent.mkdir(parents=True, exist_ok=True)
        execute(["git", "clone", OFFICIAL_URL, str(directory)], cwd=ROOT)
        execute(["git", "checkout", "--detach", OFFICIAL_SHA], cwd=directory)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=directory, text=True
    ).strip()
    if actual != OFFICIAL_SHA:
        raise RuntimeError(
            f"Upstream checkout is {actual}, not pinned {OFFICIAL_SHA}. "
            "Use a separate --official-dir (the script will not alter your checkout)."
        )
    for name in ("main.py", "ProbeGen.py", "data.py"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Missing unmodified upstream file: {directory / name}")


def link_dataset(directory: Path) -> None:
    data_dir = ROOT / "data" / "classification" / "fmnist_inr"
    real_data = (data_dir / "dataset" / "fmnist_inrs").resolve()
    if not real_data.is_dir():
        raise FileNotFoundError(
            f"{real_data} not found. Run scripts/setup_data/classification_fmnist.sh"
        )
    official_root = directory / "experiments" / "inr_classification" / "dataset"
    split_path = official_root / "fmnist_splits.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Original official split missing: {split_path}")
    splits = json.loads(split_path.read_text())
    # Use the path prefixes from the ORIGINAL split file rather than rewriting it.
    anchors = set()
    examples = []
    for split in ("train", "val", "test"):
        relative = splits[split]["path"]
        if not relative:
            raise RuntimeError(f"Original split {split} is empty")
        for value in (relative[0], relative[-1]):
            sample = Path(value)
            if sample.is_absolute() or ".." in sample.parts:
                raise ValueError(f"Unexpected absolute/traversal path in official split: {sample}")
            try:
                index = sample.parts.index("fmnist_inrs")
            except ValueError as exc:
                raise ValueError(f"Unexpected official split path: {sample}") from exc
            anchors.add(sample.parts[:index + 1])
            examples.append(sample)
    if len(anchors) != 1:
        raise RuntimeError(f"Inconsistent official split data paths: {anchors}")
    anchor = next(iter(anchors))
    link = official_root.joinpath(*anchor)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != real_data:
            raise RuntimeError(f"Existing dataset symlink {link} points elsewhere")
    elif link.exists():
        if not link.samefile(real_data):
            raise RuntimeError(f"{link} already exists and is not the expected dataset")
    else:
        link.symlink_to(real_data, target_is_directory=True)
    for sample in examples:
        if not (official_root / sample).is_file():
            raise FileNotFoundError(f"Original split checkpoint is missing: {official_root / sample}")
    print(f"[dataset] Unmodified official splits; checkpoints linked: {link} -> {real_data}")


def original_command(python: str, q: int, seed: int, experiment: str) -> list[str]:
    # Exactly the options in upstream scripts/main_results/fmnist_inr__ProbeGen_128.sh,
    # with only --n_tokens and --seed changed. Upstream main.py implements
    # Adam(lr=3e-4) + CosineAnnealingLR(T_max=30*len(train_loader)).
    return [
        python, "-u", "main.py",
        "--exp_name", experiment,
        "--seed", str(seed),
        "--dataset", "fmnist_inr",
        "--n_tokens", str(q),
        "--d_hid", "256",
        "--n_layers", "6",
        "--gen_type", "linear_2_no_acts",
        "--batch_size", "32",
        "--lr", "0.0003",
        "--epochs", "30",
        "--eval_every", "500",
        "--n_workers", "0",
        "--device", "cuda",
    ]


def best_validation_row(run_dir: Path) -> dict:
    log = run_dir / "log.csv"
    if not (run_dir / "epoch_last.pth").is_file() or not log.is_file():
        raise RuntimeError(f"Original run has no completed last checkpoint or log: {run_dir}")
    with log.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Original main.py logs val_acc and test_acc every 500 steps; no custom eval.
    valid = []
    for row in rows:
        try:
            val = float(row["val_acc"])
            test = float(row["test_acc"])
            if math.isfinite(val) and math.isfinite(test):
                valid.append((val, row, test))
        except (TypeError, KeyError, ValueError):
            pass
    if not valid:
        raise RuntimeError(f"No valid original val/test accuracy rows in {log}")
    val, row, test = max(valid, key=lambda item: item[0])
    return {
        "best_val_acc": val,
        "test_acc_at_best_val": test,
        "best_logged_step": row["global_step"],
    }


def save_tables(results_dir: Path, rows: list[dict], probes: list[int], seeds: list[int]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    per_seed = results_dir / "per_seed.csv"
    cols = ["q", "seed", "best_val_acc", "test_acc_at_best_val",
            "best_logged_step", "official_exp_dir"]
    with per_seed.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for q in probes:
        group = [r for r in rows if r["q"] == q]
        if len(group) != len(seeds):
            continue
        values = [r["test_acc_at_best_val"] for r in group]
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append({
            "q": q,
            "n_seeds": len(values),
            "test_acc_mean": statistics.mean(values),
            "test_acc_std": sd,
            "test_acc_sem": sd / math.sqrt(len(values)),
        })
    with (results_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "q", "n_seeds", "test_acc_mean", "test_acc_std", "test_acc_sem"
        ])
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(f'[summary] original ProbeGen Q={row["q"]:3d}: '
              f'test_acc={row["test_acc_mean"]:.5f} '
              f'+/- {row["test_acc_sem"]:.5f} SEM ({row["n_seeds"]} seeds)', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--official-dir", type=Path,
                        default=ROOT / "third_party" / "ProbeGen_official")
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "checkpoints" / "probe_count_original" / "fmnist_inr")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (not args.probes or any(q <= 0 for q in args.probes)
            or len(set(args.probes)) != len(args.probes)
            or not args.seeds or any(s < 0 for s in args.seeds)
            or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Probes must be positive and seeds nonnegative, with no duplicates")

    official = args.official_dir.expanduser().resolve()
    results = args.results_dir.expanduser().resolve()
    python = sys.executable  # The current conda environment, e.g. yon.
    for q in args.probes:
        for seed in args.seeds:
            name = f"original_probegen_fmnist_Q{q}_s{seed}"
            cmd = original_command(python, q, seed, name)
            if args.dry_run:
                print(f"[DRY RUN] cwd={official} " + shlex.join(cmd))
    if args.dry_run:
        return

    if not args.skip_setup:
        execute(["bash", "scripts/setup_data/classification_fmnist.sh"], cwd=ROOT)
    prepare_upstream(official)
    link_dataset(official)

    rows = []
    for q in args.probes:
        for seed in args.seeds:
            name = f"original_probegen_fmnist_Q{q}_s{seed}"
            run_dir = official / "experiments" / "fmnist_inr" / "runs" / name
            if (run_dir / "epoch_last.pth").is_file() and (run_dir / "log.csv").is_file():
                print(f"[REUSE] {run_dir}")
            else:
                execute(original_command(python, q, seed, name), cwd=official,
                        log=results / "logs" / f"{name}.log")
            row = best_validation_row(run_dir)
            rows.append({"q": q, "seed": seed, **row, "official_exp_dir": str(run_dir)})
            save_tables(results, rows, args.probes, args.seeds)
    print(f"[DONE] Official code SHA {OFFICIAL_SHA}; results: {results}", flush=True)


if __name__ == "__main__":
    main()
