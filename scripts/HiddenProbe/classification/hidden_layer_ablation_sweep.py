#!/usr/bin/env python3
"""Validation-based sweep + 5-seed evaluation for one HiddenProbe INR layer.

Invoked by the four run_*_layer{0,1}_sweep.sh wrappers.
FMNIST: 24 plateau + 3 cosine configs, 20 sweep epochs, 30 final epochs.
CIFAR-10 non-aug: the same 27-config grid, 20 sweep epochs,
60 final epochs to match run_cifar10_inr_nonaug.sh.

The two training backends retain their respective model architectures.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LRS = ("3e-4", "5e-4", "7e-4")
PATIENCES = (3, 5)
FACTORS = ("0.2", "0.3", "0.5", "0.7")


@dataclass(frozen=True)
class Config:
    lr: str
    scheduler: str
    patience: int | None = None
    factor: str | None = None

    @property
    def tag(self) -> str:
        if self.scheduler == "cosine":
            return f"lr{self.lr}_cosine"
        return f"lr{self.lr}_plateau_pat{self.patience}_fac{self.factor}"

    def to_dict(self) -> dict:
        return {
            "lr": self.lr,
            "scheduler": self.scheduler,
            "plateau_patience": self.patience,
            "plateau_factor": self.factor,
        }


def configs(dataset: str) -> list[Config]:
    result = [
        Config(lr, "plateau", patience, factor)
        for lr, patience, factor in itertools.product(LRS, PATIENCES, FACTORS)
    ]
    result.extend(Config(lr, "cosine") for lr in LRS)
    return result


def read_summary(dataset: str, run_dir: Path, seed: int) -> dict | None:
    """Only complete, finite validation/test records count as completed."""
    try:
        if dataset == "fmnist":
            with (run_dir / "seeds_summary.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 1:
                return None
            record = rows[0]
        else:
            with (run_dir / "summary.json").open() as stream:
                record = json.load(stream)

        if int(record["seed"]) != seed:
            return None
        val_acc = float(record["best_val_acc"])
        test_acc = float(record["best_test_acc"])
        if not math.isfinite(val_acc) or not math.isfinite(test_acc):
            return None
        return {"val_acc": val_acc, "test_acc": test_acc}
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def experiment_dir(dataset: str, exp_name: str, cifar_runs_dir: Path) -> Path:
    if dataset == "fmnist":
        return ROOT / "experiments/classification/fmnist/runs" / exp_name
    return cifar_runs_dir / exp_name


def build_command(
    python: str,
    dataset: str,
    layer: int,
    config: Config,
    epochs: int,
    seed: int,
    exp_name: str,
    cifar_runs_dir: Path,
) -> list[str]:
    command = [
        python, "main.py",
        "--method", "hiddenprobe",
        "--task", "classification",
        "--dataset", dataset,
        "--include_hidden_features", "true",
        "--hidden_layer_ids", str(layer),
        "--exp_name", exp_name,
        "--seed", str(seed),
        "--n_probes", "128",
        "--epochs", str(epochs),
        "--lr", config.lr,
        "--scheduler", config.scheduler,
    ]

    if config.scheduler == "plateau":
        command.extend([
            "--plateau_patience", str(config.patience),
            "--plateau_factor", str(config.factor),
        ])

    if dataset == "fmnist":
        # Preserve the canonical FMNIST HiddenProbe architecture.
        command.extend([
            "--num_seeds", "1",
            "--batch_size", "64",
            "--d_hid", "256",
            "--mixer_n_layers", "6",
            "--gen_type", "linear_2_no_acts",
            "--gen_latent_z", "32",
            "--generator_width", "16",
            "--per_probe_mlp", "mlp2",
            "--per_probe_mlp_width", "256",
            "--per_probe_out_dim", "4",
            "--per_probe_init", "standard",
            "--r_per_hidden", "2",
            "--rank", "8",
            "--probe_lr", config.lr,
            "--weight_decay", "0.0",
            "--plateau_monitor", "val_acc",
            "--plateau_min_lr", "1e-6",
            "--eval_every", "500",
            "--n_workers", "0",
            "--device", os.environ.get("DEVICE", "cuda"),
        ])
    else:
        # Preserve the actual non-augmented CIFAR HiddenProbe head and
        # probe-optimizer settings from run_cifar10_inr_nonaug.sh.
        command.extend([
            "--gen_type", "linear_2_no_acts",
            "--gen_latent_z", "32",
            "--generator_width", "16",
            "--domain_tanh", "1",
            "--head", "set_transformer",
            "--d", "120",
            "--nenc", "2",
            "--nheads", "8",
            "--ema_decay", "0.999",
            "--probe_lr", "7e-4",
            "--batch_size", "32",
            "--warmup", "300",
            "--dropout", "0.1",
            "--head_wd", "0.1",
            "--plateau_min_lr", "1e-6",
            "--eval_every", "500",
            "--runs_dir", str(cifar_runs_dir),
        ])
    return command


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fmnist", "cifar10"), required=True)
    parser.add_argument("--layer-id", choices=(0, 1), type=int, required=True)
    parser.add_argument("--python", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--sweep-epochs", type=int, default=int(os.getenv("SWEEP_EPOCHS", "20")))
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=int(os.getenv("NUM_SEEDS", "5")))
    parser.add_argument("--dry-run", action="store_true",
                        help="Show planned configurations/commands; do not train.")
    args = parser.parse_args()

    if args.final_epochs is None:
        args.final_epochs = int(
            os.getenv("FINAL_EPOCHS", "30" if args.dataset == "fmnist" else "60")
        )
    if min(args.sweep_epochs, args.final_epochs, args.num_seeds) <= 0:
        parser.error("Epoch counts and number of seeds must be positive.")

    layer = args.layer_id
    prefix = f"hiddenprobe_{args.dataset}_hiddenL{layer}"
    work_dir = ROOT / "checkpoints/hidden_layer_ablation" / args.dataset / f"layer{layer}"
    cifar_runs_dir = work_dir / "runs"
    sweep_root = work_dir / f"sweep_{args.sweep_epochs}epochs"
    final_root = work_dir / f"final_{args.final_epochs}epochs"
    sweep_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)
    choices = configs(args.dataset)
    print(f"[SWEEP] dataset={args.dataset} hidden_layer_id={layer} "
          f"configs={len(choices)} sweep_epochs={args.sweep_epochs} "
          f"final_epochs={args.final_epochs} final_seeds={args.num_seeds}",
          flush=True)

    rows: list[dict] = []
    for i, cfg in enumerate(choices, 1):
        exp_name = f"{prefix}_SWEEP{args.sweep_epochs}_{cfg.tag}_s0"
        run_dir = experiment_dir(args.dataset, exp_name, cifar_runs_dir)
        command = build_command(
            args.python, args.dataset, layer, cfg, args.sweep_epochs, 0,
            exp_name, cifar_runs_dir
        )
        print(f"[SWEEP {i}/{len(choices)}] {cfg.tag}", flush=True)
        record = read_summary(args.dataset, run_dir, 0)
        if args.dry_run:
            print("  " + shlex.join(command), flush=True)
            continue
        if record is None:
            # An interrupted run is restarted cleanly; never reuse partial
            # checkpoints or stale log.csv from the same experiment.
            if run_dir.exists():
                shutil.rmtree(run_dir)
            subprocess.run(command, cwd=ROOT, check=True)
            record = read_summary(args.dataset, run_dir, 0)
            if record is None:
                raise RuntimeError(f"Missing/invalid sweep result: {run_dir}")
        else:
            print("  [skip] Completed sweep configuration", flush=True)
        rows.append({
            **cfg.to_dict(),
            "layer_id": layer,
            "seed": 0,
            "best_val_acc": record["val_acc"],
            "test_acc_logged_not_used_for_selection": record["test_acc"],
            "exp_name": exp_name,
        })

    if args.dry_run:
        print(f"[DRY RUN] planned {len(choices)} sweep configs + "
              f"{args.num_seeds} final seeds; no work executed.")
        return

    rows.sort(key=lambda row: row["best_val_acc"], reverse=True)
    write_csv(
        sweep_root / "sweep_results.csv", rows,
        ["lr", "scheduler", "plateau_patience", "plateau_factor",
         "layer_id", "seed", "best_val_acc",
         "test_acc_logged_not_used_for_selection", "exp_name"],
    )
    winner = rows[0]
    (sweep_root / "selected_config.json").write_text(
        json.dumps(winner, indent=2) + "\n"
    )
    print(f"[SELECTED BY VALIDATION] {winner}", flush=True)
    chosen = Config(
        lr=winner["lr"],
        scheduler=winner["scheduler"],
        patience=winner["plateau_patience"],
        factor=winner["plateau_factor"],
    )

    final_rows: list[dict] = []
    for seed in range(args.num_seeds):
        exp_name = (
            f"{prefix}_FINAL{args.final_epochs}_{chosen.tag}_s{seed}"
        )
        run_dir = experiment_dir(args.dataset, exp_name, cifar_runs_dir)
        record = read_summary(args.dataset, run_dir, seed)
        if record is None:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            command = build_command(
                args.python, args.dataset, layer, chosen, args.final_epochs,
                seed, exp_name, cifar_runs_dir,
            )
            print(f"[FINAL {seed + 1}/{args.num_seeds}] {chosen.tag} seed={seed}",
                  flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            record = read_summary(args.dataset, run_dir, seed)
            if record is None:
                raise RuntimeError(f"Missing/invalid final result: {run_dir}")
        else:
            print(f"[FINAL] [skip] completed seed={seed}", flush=True)
        final_rows.append({
            "dataset": args.dataset,
            "layer_id": layer,
            "seed": seed,
            "best_val_acc": record["val_acc"],
            "best_test_acc": record["test_acc"],
            "exp_name": exp_name,
        })

    write_csv(
        final_root / "five_seeds.csv", final_rows,
        ["dataset", "layer_id", "seed", "best_val_acc", "best_test_acc", "exp_name"],
    )
    test_accs = [row["best_test_acc"] for row in final_rows]
    val_accs = [row["best_val_acc"] for row in final_rows]
    summary = {
        "dataset": args.dataset,
        "layer_id": layer,
        "selected_by": "best_val_acc",
        "selected_config": chosen.to_dict(),
        "sweep_epochs": args.sweep_epochs,
        "final_epochs": args.final_epochs,
        "num_seeds": args.num_seeds,
        "val_acc_mean": statistics.mean(val_accs),
        "test_acc_mean": statistics.mean(test_accs),
        "test_acc_std": statistics.stdev(test_accs) if len(test_accs) > 1 else 0.0,
    }
    (final_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[DONE] Test acc = {summary['test_acc_mean']:.6f} +/- "
          f"{summary['test_acc_std']:.6f}; results: {final_root}",
          flush=True)


if __name__ == "__main__":
    main()
