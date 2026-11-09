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
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from models.logging_utils import print_run_config, print_eval, print_seed_result

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_URL = "https://github.com/jonkahana/ProbeGen.git"
# Pinned GitHub main, verified 2026-11-09. Do not silently follow upstream changes.
OFFICIAL_SHA = "e3b8e1485b01dcbccb0c508896d9c2d101a34fa3"


def execute(cmd: list[str], *, cwd: Path) -> None:
    """Launch non-training shell commands with their normal terminal output."""
    print("+", shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def original_parameter_count(q: int) -> int:
    """Exact original ProbeGen.py formula for linear_2_no_acts, width=256, six layers."""
    latent, width, num_classes = 32, 256, 10
    return (
        q * latent  # Learned latent probes.
        + (latent * latent + latent)  # 32 -> 32 generator.
        + (2 * latent + 2)  # 32 -> 2 generator.
        + (q * width + width)  # First mixer layer: Q -> 256.
        + 4 * (width * width + width)  # Four 256 -> 256 mixers.
        + (width * num_classes + num_classes)  # Output: 256 -> 10.
    )


def official_split_sizes(directory: Path) -> dict[str, int]:
    path = directory / "experiments" / "inr_classification" / "dataset" / "fmnist_splits.json"
    split = json.loads(path.read_text())
    return {name: len(split[name]["label"]) for name in ("train", "val", "test")}


def show_live_evaluations(
    log_csv: Path, *, seen_steps: set[int], start_wall_time: float,
    total_steps: int,
) -> None:
    """Watch the official main.py CSV; display only, never intervene in training."""
    if not log_csv.is_file() or log_csv.stat().st_mtime < start_wall_time - 0.1:
        return
    try:
        with log_csv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, UnicodeError, csv.Error):
        return
    # The official training code rewrites the complete CSV after each evaluation.
    # Read the whole file so the validation best is correct even if several
    # evaluations finish between two polling intervals.
    valid = []
    for row in rows:
        try:
            step = int(row["global_step"])
            val = float(row["val_acc"])
            test = float(row["test_acc"])
            loss = float(row["train_loss"])
            epoch = int(row["epoch"]) + 1
            if not all(map(math.isfinite, (val, test, loss))):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        valid.append((step, epoch, loss, val, test))
    best_val = -math.inf
    for step, epoch, loss, val, test in sorted(valid):
        is_best = val > best_val
        best_val = max(best_val, val)
        if step in seen_steps:
            continue
        seen_steps.add(step)
        elapsed = max(0.0, time.time() - start_wall_time)
        remaining = (elapsed / max(step, 1)) * max(total_steps - step, 0)
        print_eval(
            task="classification", step=step, epoch=epoch,
            train_loss=loss, val_value=val, test_value=test,
            elapsed=elapsed, remaining=remaining, new_best=is_best,
        )


def run_official_with_live_display(
    cmd: list[str], *, official: Path, experiment: str, q: int,
    seed: int, split_sizes: dict[str, int], results_dir: Path,
) -> None:
    """Render the same RUN CONFIG / EVAL format as the other project runners.

    The model and optimizer run in the pinned official script, unmodified.
    The official tqdm/raw output is saved separately to keep terminal logging
    clean and comparable with the other HiddenProbe backends.
    """
    print_run_config(
        method="ProbeGen (original)", task="classification", dataset="FMNIST-INR",
        seed=seed, experiment=experiment,
        train_size=split_sizes["train"], val_size=split_sizes["val"],
        test_size=split_sizes["test"], probes=q,
        parameters=original_parameter_count(q),
        trainable=original_parameter_count(q), device="cuda",
    )
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_log = log_dir / f"{experiment}.log"
    run_dir = official / "experiments" / "fmnist_inr" / "runs" / experiment
    log_csv = run_dir / "log.csv"
    start_wall_time = time.time()
    total_steps = 30 * math.ceil(split_sizes["train"] / 32)
    seen_steps: set[int] = set()
    print(f"[upstream] Original ProbeGen.py / CosineAnnealingLR; raw output: {raw_log}", flush=True)
    with raw_log.open("w") as output:
        with subprocess.Popen(
            cmd, cwd=official, stdout=output, stderr=subprocess.STDOUT,
        ) as proc:
            while proc.poll() is None:
                show_live_evaluations(log_csv, seen_steps=seen_steps,
                                      start_wall_time=start_wall_time,
                                      total_steps=total_steps)
                time.sleep(0.5)
            status = proc.returncode
    show_live_evaluations(log_csv, seen_steps=seen_steps,
                          start_wall_time=start_wall_time, total_steps=total_steps)
    if status:
        last = raw_log.read_text(errors="replace").splitlines()[-35:]
        print("\n".join(last), file=sys.stderr, flush=True)
        raise RuntimeError(
            f"Original ProbeGen process failed (exit {status}); full log: {raw_log}"
        )


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
        "best_logged_epoch": int(row["epoch"]) + 1,
    }


def save_tables(results_dir: Path, rows: list[dict], probes: list[int], seeds: list[int]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    per_seed = results_dir / "per_seed.csv"
    cols = ["q", "seed", "best_val_acc", "test_acc_at_best_val",
            "best_logged_step", "best_logged_epoch", "official_exp_dir"]
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

    split_sizes = official_split_sizes(official)
    rows = []
    for q in args.probes:
        for seed in args.seeds:
            name = f"original_probegen_fmnist_Q{q}_s{seed}"
            run_dir = official / "experiments" / "fmnist_inr" / "runs" / name
            if (run_dir / "epoch_last.pth").is_file() and (run_dir / "log.csv").is_file():
                print(f"[REUSE] {run_dir}")
            else:
                run_official_with_live_display(
                    original_command(python, q, seed, name),
                    official=official, experiment=name, q=q,
                    seed=seed, split_sizes=split_sizes, results_dir=results,
                )
            row = best_validation_row(run_dir)
            if "best_logged_epoch" in row:
                print_seed_result(
                    task="classification", seed=seed,
                    best_epoch=row["best_logged_epoch"],
                    best_step=row["best_logged_step"],
                    val_value=row["best_val_acc"],
                    test_value=row["test_acc_at_best_val"],
                )
            rows.append({"q": q, "seed": seed, **row, "official_exp_dir": str(run_dir)})
            save_tables(results, rows, args.probes, args.seeds)
    print(f"[DONE] Official code SHA {OFFICIAL_SHA}; results: {results}", flush=True)


if __name__ == "__main__":
    main()
