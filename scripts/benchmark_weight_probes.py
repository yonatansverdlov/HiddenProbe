#!/usr/bin/env python3
"""Time one real sweep configuration per benchmark and baseline, without downloads.

Example (FMNIST-INR only, ProbeX and MVProbe simultaneously):
    python scripts/benchmark_weight_probes.py --datasets fmnist_inr --parallel 2 --epochs 60

This script never calls the data download scripts and never starts a full sweep.
It creates the MNIST-Transformers *local cache* from already downloaded checkpoints
if it is absent; cache construction is excluded from training timing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = (
    ("mnist_transformers", "MNIST-Transformers", 1728, "regression",
     "models/mvprobe_transformer_trainer.py"),
    ("svhn", "SVHN", 864, "regression", "models/mvprobe_svhn_trainer.py"),
    ("mnist_regression", "MNIST regression", 864, "regression",
     "models/mvprobe_svhn_trainer.py"),
    ("fmnist", "FMNIST INR", 576, "classification",
     "models/mvprobe_inr_trainer.py"),
    ("cifar10", "CIFAR-10 INR", 576, "classification",
     "models/mvprobe_inr_trainer.py"),
)


def secs_hms(s: str) -> float:
    h, m, sec = (int(p) for p in s.split(":"))
    return float(h * 3600 + m * 60 + sec)


def check_local_data(root: Path, datasets: set[str]) -> None:
    required = {
        "svhn": [
            root / "regression/svhn/weights.npy",
            root / "regression/svhn/layout.csv",
            root / "regression/svhn/metrics.csv.gz",
            root / "regression/svhn/split.csv",
        ],
        "mnist_regression": [
            root / "regression/mnist/weights.npy",
            root / "regression/mnist/layout.csv",
            root / "regression/mnist/metrics.csv.gz",
            root / "regression/mnist/split.csv",
        ],
        "fmnist": [root / "classification/fmnist_inr/fmnist_splits.json"],
        "cifar10": [root / "classification/cifar10_inr/nfn_cifar_split_noaug.json"],
    }
    missing = [str(p) for name, items in required.items() if name in datasets
               for p in items if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Dataset files are missing; no downloads will be attempted:\n  "
            + "\n  ".join(missing)
        )


def ensure_transformer_cache(root: Path) -> None:
    caches = [
        root / f"tpcache_mnist_{split}_cut00_s0.pt"
        for split in ("train", "val")
    ]
    if all(p.is_file() for p in caches):
        print("[setup] MNIST-Transformers cache already present.", flush=True)
        return

    source = root / "mnist_transformer"
    if not source.is_dir() or not any(source.rglob("*_75_*.pt")):
        raise FileNotFoundError(
            f"Missing extracted MNIST-Transformers epoch-75 checkpoints under {source}. "
            "This benchmarking script does not download data."
        )
    print("[setup] Building local Transformer cache (excluded from timing).", flush=True)
    cmd = [sys.executable, "main.py", "transformer", "cache",
           "--dataset", "mnist", "--seed", "0", "--cut_off", "0",
           "--data_root", str(root)]
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not all(p.is_file() for p in caches):
        raise RuntimeError(f"Transformer cache command completed but missing: {caches}")


def command_for(dataset: str, method: str, trainer: str, data_root: Path,
                output: Path, epochs: int, device: str) -> list[str]:
    common = [
        sys.executable, trainer,
        "--model_variant", method,
        "--layer_index", "0",
        "--n_probes", "128",
        "--proj_dim", "128",
        "--rep_dim", "512",
        "--lr", "3e-4",
        "--weight_decay", "1e-5",
        "--batch_size", "128",
        "--scheduler", "plateau",
        "--plateau_factor", "0.5",
        "--plateau_patience", "5",
        "--plateau_min_lr", "1e-6",
        "--epochs", str(epochs),
        "--seed", "0",
        "--device", device,
        "--skip_test_eval",
        "--out_dir", str(output),
    ]
    if dataset == "mnist_transformers":
        return common + ["--data_dir", str(data_root),
                         "--dataset_name", "MNIST-Transformers"]
    if dataset in {"svhn", "mnist_regression"}:
        ds = "svhn" if dataset == "svhn" else "mnist"
        return common + [
            "--data_dir", str(data_root / "regression" / ds),
            "--split_csv", str(data_root / "regression" / ds / "split.csv"),
            "--dataset_name", "SVHN" if ds == "svhn" else "MNIST",
        ]
    if dataset == "fmnist":
        return common + [
            "--data_dir", str(data_root / "classification/fmnist_inr"),
            "--split_json", str(data_root / "classification/fmnist_inr/fmnist_splits.json"),
            "--dataset_name", "Fashion-MNIST INR",
            "--n_classes", "10",
        ]
    return common + [
        "--data_dir", str(data_root / "classification/cifar10_inr"),
        "--split_json", str(data_root / "classification/cifar10_inr/nfn_cifar_split_noaug.json"),
        "--dataset_name", "CIFAR-10 INR non-aug",
        "--n_classes", "10",
    ]


def measure(spec: tuple, method: str, data_root: Path, output: Path,
            epochs: int, device: str) -> dict:
    dataset, display, n_configs, task, trainer = spec
    folder = output / f"{method}_{dataset}"
    folder.mkdir(parents=True, exist_ok=True)
    logfile = folder / "training.log"
    command = command_for(dataset, method, trainer, data_root, folder, epochs, device)
    print(f"[start] {method:7s} {display:20s} | {epochs} epochs", flush=True)
    start = time.monotonic()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with logfile.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    wall = time.monotonic() - start
    last_elapsed = re.findall(
        r"EVAL\s*\|[^\n]*\belapsed=(\d{2}:\d{2}:\d{2})", 
        logfile.read_text(encoding="utf-8", errors="replace")
    )
    training_sec = secs_hms(last_elapsed[-1]) if last_elapsed else None
    if training_sec is not None and training_sec <= 0:
        training_sec = None
    row = {
        "method": method,
        "dataset": dataset,
        "configs_in_full_sweep": n_configs,
        "epochs": epochs,
        "status": "ok" if result.returncode == 0 and (folder / "summary.json").is_file()
                  else f"FAILED (exit {result.returncode})",
        "wall_seconds": round(wall, 2),
        "training_seconds": training_sec,
        "initialization_seconds_approx": round(max(0, wall - training_sec), 2)
                                         if training_sec is not None else "",
        "estimated_full_sweep_job_hours": round(
            (training_sec if training_sec is not None else wall)
            * n_configs * 60 / epochs / 3600, 2
        ),
        "log": str(logfile.relative_to(ROOT)) if logfile.is_relative_to(ROOT) else str(logfile),
    }
    print(f"[done]  {method:7s} {display:20s} | {row['status']} | "
          f"wall {wall:.1f}s | training "
          f"{training_sec if training_sec is not None else 'n/a'}s",
          flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=("probex", "mvprobe"),
                        default=("probex", "mvprobe"))
    parser.add_argument("--parallel", type=int, default=5,
                        help="Maximum concurrent independent training processes")
    parser.add_argument("--datasets", nargs="+",
                        choices=[item[0] for item in BENCHMARKS] + ["fmnist_inr"],
                        default=[item[0] for item in BENCHMARKS],
                        help="Subset of benchmark datasets; e.g. --datasets fmnist_inr")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Training epochs per single config (default 60)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("DATA_ROOT", ROOT / "data")))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "checkpoints/weight_probe_timing")
    args = parser.parse_args()
    if args.parallel <= 0 or args.epochs <= 0:
        parser.error("--parallel and --epochs must be positive")
    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # None of the methods below downloads data. Cache building is local and
    # excluded from timing, matching the assumption that the zoo is present.
    chosen = {"fmnist" if name == "fmnist_inr" else name for name in args.datasets}
    selected = [spec for spec in BENCHMARKS if spec[0] in chosen]
    check_local_data(data_root, chosen)
    if "mnist_transformers" in chosen:
        ensure_transformer_cache(data_root)

    rows = []
    jobs = [(spec, method) for spec in selected for method in args.methods]
    print(f"\n=== Timing {len(jobs)} configurations using up to "
          f"{args.parallel} concurrent processes ===", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(measure, spec, method, data_root,
                               output, args.epochs, args.device)
                   for spec, method in jobs]
        for future in as_completed(futures):
            rows.append(future.result())

    out_csv = output / "timing_results.csv"
    cols = ["method", "dataset", "configs_in_full_sweep", "epochs", "status",
            "wall_seconds", "training_seconds", "initialization_seconds_approx",
            "estimated_full_sweep_job_hours", "log"]
    with out_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: (x["method"], x["dataset"])))

    print("\n=== TRAINING-ONLY SWEEP ESTIMATES ===")
    for row in sorted(rows, key=lambda x: (x["method"], x["dataset"])):
        print(f"{row['method']:7s} {row['dataset']:19s} "
              f"{row['status']:18s}  "
              f"full grid ~{row['estimated_full_sweep_job_hours']:9.2f} "
              "job-hours")
    if all(row["status"] == "ok" for row in rows):
        total = sum(row["estimated_full_sweep_job_hours"] for row in rows)
        print(f"TOTAL: ~{total:.2f} job-hours; "
              f"rough wall-clock at parallel={args.parallel}: "
              f"~{total/args.parallel:.2f} hours.")
        print("WARNING: This assumes similar throughput under a full sweep; "
              "GPU contention, layer sizes, and I/O can change timing.")
    else:
        print("Some jobs failed: inspect their training.log files. "
              "Failed timings are not reliable estimates.")
    print(f"Results saved to {out_csv}")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
