#!/usr/bin/env python3
"""Transformer single-block ablation: validation-only 27-config sweep + 5-seed final.

For each of MNIST-Transformers / AGNews-Transformers and Block 1 / Block 2,
the model observes exactly: selected block, final teacher LayerNorm, logits.
The two ablations use the same channel-trajectory readout, generator, probe
bank geometry, capacity and hyperparameter grid. Original baselines are intact.

Grid mirrors the existing HiddenProbe INR layer ablation:
  pred_lr in {3e-4, 5e-4, 7e-4}
  plateau: patience {3,5}, factor {0.2,0.3,0.5,0.7}
  cosine: one per pred_lr = 27 configs.

Unlike INR experiments, Transformers use optimizer updates, not epochs:
default sweep and final budgets are 40,000 updates as in the original
MNIST/AGNews Transformer baseline runners. Both budgets are configurable.
Validation Kendall tau alone selects the winner. Test tau is used only
for final reporting after choosing hyperparameters.
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
DATASET_SETTINGS = {
    "mnist": {"n_classes": 10, "gen_lr": "5e-4", "ffn": 384},
    "agnews": {"n_classes": 4, "gen_lr": "1e-3", "ffn": 384},
}


@dataclass(frozen=True)
class Config:
    pred_lr: str
    scheduler: str
    patience: int = 5
    factor: str = "0.5"

    @property
    def tag(self) -> str:
        if self.scheduler == "cosine":
            return f"lr{self.pred_lr}_cosine"
        return f"lr{self.pred_lr}_plateau_pat{self.patience}_fac{self.factor}"

    def as_dict(self) -> dict:
        return {
            "pred_lr": self.pred_lr,
            "scheduler": self.scheduler,
            "plateau_patience": self.patience if self.scheduler == "plateau" else None,
            "plateau_factor": self.factor if self.scheduler == "plateau" else None,
        }


def configs() -> list[Config]:
    result = [
        Config(lr, "plateau", p, f)
        for lr, p, f in itertools.product(LRS, PATIENCES, FACTORS)
    ]
    result.extend(Config(lr, "cosine") for lr in LRS)
    assert len(result) == 27 and len({c.tag for c in result}) == 27
    return result


def build_command(args, cfg: Config, seed: int, budget: int, run_dir: Path) -> list[str]:
    settings = DATASET_SETTINGS[args.dataset]
    return [
        args.python, str(ROOT / "main.py"), "transformer", "train",
        "--method", "hiddenprobe",
        "--dataset", args.dataset,
        "--generator", "g3",
        "--n_classes", str(settings["n_classes"]),
        "--ffn", str(settings["ffn"]),
        "--n_probes", "256",
        "--readout", "multi",
        "--readout_arch", "channel_trajectory",
        "--hidden_block_only", str(args.block),
        "--cut_off", "0",
        "--pred_lr", cfg.pred_lr,
        "--gen_lr", settings["gen_lr"],
        "--scheduler", cfg.scheduler,
        "--plateau_patience", str(cfg.patience),
        "--plateau_factor", cfg.factor,
        "--warmup", "0",
        "--weight_decay", "1e-3",
        "--dropout", "0.1",
        "--max_updates", str(budget),
        "--eval_every", "1000",
        "--micro", "32",
        "--seed", str(seed),
        "--exact_accum",
        "--data_root", str(args.data_root),
        "--runs_dir", str(run_dir),
    ]


def inspect_checkpoint(run_dir: Path, args, cfg: Config, seed: int, budget: int):
    checkpoint = run_dir / "last.pt"
    if not checkpoint.exists():
        return None
    import torch

    try:
        obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
        a = obj["cfg"]
        required = {
            "dataset": args.dataset,
            "method": "hiddenprobe",
            "readout_arch": "channel_trajectory",
            "hidden_block_only": args.block,
            "seed": seed,
            "n_probes": 256,
            "max_updates": budget,
            "scheduler": cfg.scheduler,
        }
        for key, expected in required.items():
            if a.get(key) != expected:
                raise ValueError(
                    f"{checkpoint}: incompatible {key}, got {a.get(key)!r}, expected {expected!r}"
                )
        if not math.isclose(float(a["pred_lr"]), float(cfg.pred_lr), rel_tol=1e-10):
            raise ValueError("pred_lr mismatch in checkpoint")
        if cfg.scheduler == "plateau" and (
            a["plateau_patience"] != cfg.patience
            or not math.isclose(float(a["plateau_factor"]), float(cfg.factor), rel_tol=1e-10)
        ):
            raise ValueError("scheduler hyperparameter mismatch in checkpoint")
        val = float(obj["best_val_tau"])
        test = float(obj["best_test_tau"])
        if not (math.isfinite(val) and math.isfinite(test)):
            raise ValueError("Non-finite score")
        return {"val_tau": val, "test_tau": test}
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, EOFError) as error:
        raise RuntimeError(
            f"Invalid or incompatible completed checkpoint {checkpoint}: {error}. "
            "Do not overwrite; inspect or move it aside."
        ) from error


def execute(cmd: list[str], log_file: Path) -> None:
    print("  " + shlex.join(cmd), flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", buffering=1) as fh:
        with subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                fh.write(line)
            rc = proc.wait()
    if rc:
        raise RuntimeError(f"Training failed with exit {rc}. See {log_file}")


def run_or_reuse(args, cfg: Config, seed: int, budget: int, run_dir: Path, log_file: Path):
    existing = inspect_checkpoint(run_dir, args, cfg, seed, budget)
    if existing is not None:
        print(f"  [SKIP] completed: {run_dir}", flush=True)
        return existing
    if run_dir.exists():
        # No completed best-validation checkpoint: restart this configuration.
        shutil.rmtree(run_dir)
    execute(build_command(args, cfg, seed, budget, run_dir), log_file)
    result = inspect_checkpoint(run_dir, args, cfg, seed, budget)
    if result is None:
        raise RuntimeError(f"Training produced no last.pt: {run_dir}")
    return result


def csv_out(path: Path, records: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=tuple(DATASET_SETTINGS), required=True)
    p.add_argument("--block", choices=(1, 2), type=int, required=True)
    p.add_argument("--python", default=os.environ.get("PYTHON_BIN", sys.executable))
    p.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", "data")))
    p.add_argument("--sweep-updates", type=int, default=int(os.environ.get("SWEEP_UPDATES", "40000")))
    p.add_argument("--final-updates", type=int, default=int(os.environ.get("FINAL_UPDATES", "40000")))
    p.add_argument("--num-seeds", type=int, default=int(os.environ.get("NUM_SEEDS", "5")))
    p.add_argument("--only-sweep", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if min(args.sweep_updates, args.final_updates, args.num_seeds) < 1:
        p.error("Budgets and seed count must be positive")
    args.data_root = args.data_root.resolve()

    base = ROOT / "checkpoints/transformer_hidden_layer_ablation" / args.dataset / f"block{args.block}"
    sw = base / f"sweep_{args.sweep_updates}updates"
    final = base / f"final_{args.final_updates}updates"
    choices = configs()
    print(
        f"[SWEEP] {args.dataset} block={args.block} + final norm + logits; "
        f"{len(choices)} configs; sweep={args.sweep_updates} updates; "
        f"final={args.final_updates} updates x {args.num_seeds} seeds",
        flush=True,
    )
    rows = []
    for i, cfg in enumerate(choices, 1):
        run = sw / "runs" / cfg.tag
        print(f"[SWEEP {i}/{len(choices)}] {cfg.tag}", flush=True)
        if args.dry_run:
            print("  " + shlex.join(build_command(args, cfg, 0, args.sweep_updates, run)))
            continue
        result = run_or_reuse(
            args, cfg, 0, args.sweep_updates, run, sw / "logs" / f"{cfg.tag}.log"
        )
        rows.append({**cfg.as_dict(), "seed": 0, "best_val_tau": result["val_tau"],
                     "run_dir": str(run)})
    if args.dry_run:
        print("[DRY RUN] no dataset setup, training or files written.")
        return

    rows.sort(key=lambda row: row["best_val_tau"], reverse=True)
    csv_out(sw / "sweep_results.csv", rows,
            ["pred_lr", "scheduler", "plateau_patience", "plateau_factor",
             "seed", "best_val_tau", "run_dir"])
    best = rows[0]
    chosen = Config(best["pred_lr"], best["scheduler"],
                    best["plateau_patience"] or 5, best["plateau_factor"] or "0.5")
    (sw / "selected_config.json").write_text(
        json.dumps({"dataset": args.dataset, "block": args.block,
                    "selection_metric": "validation_kendall_tau", **chosen.as_dict(),
                    "best_val_tau": best["best_val_tau"], "sweep_updates": args.sweep_updates},
                   indent=2) + "\n"
    )
    print(f"[SELECTED BY VAL KENDALL TAU] {chosen.tag}: {best['best_val_tau']:.6f}", flush=True)
    if args.only_sweep:
        print(f"[DONE] sweep only; selected config: {sw / 'selected_config.json'}")
        return

    finals = []
    for seed in range(args.num_seeds):
        run = final / "runs" / chosen.tag / f"seed_{seed}"
        print(f"[FINAL {seed + 1}/{args.num_seeds}] {chosen.tag} seed={seed}", flush=True)
        result = run_or_reuse(
            args, chosen, seed, args.final_updates, run,
            final / "logs" / f"{chosen.tag}_seed{seed}.log"
        )
        finals.append({"dataset": args.dataset, "block": args.block, "seed": seed,
                       "best_val_tau": result["val_tau"], "best_test_tau": result["test_tau"],
                       "run_dir": str(run)})
    csv_out(final / "five_seeds.csv", finals,
            ["dataset", "block", "seed", "best_val_tau", "best_test_tau", "run_dir"])
    vals = [row["best_test_tau"] for row in finals]
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    summary = {
        "dataset": args.dataset,
        "block": args.block,
        "input_views": [f"block{args.block}", "final_norm", "logits"],
        "selection_metric": "validation_kendall_tau",
        "selected_config": chosen.as_dict(),
        "sweep_updates": args.sweep_updates,
        "final_updates": args.final_updates,
        "num_seeds": len(vals),
        "test_tau_mean": statistics.mean(vals),
        "test_tau_sd": sd,
        "test_tau_se": sd / math.sqrt(len(vals)),
    }
    final.mkdir(parents=True, exist_ok=True)
    (final / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[DONE] Test tau = {summary['test_tau_mean']:.6f} +/- "
        f"{summary['test_tau_se']:.6f} (SE); results: {final}",
        flush=True,
    )


if __name__ == "__main__":
    main()
