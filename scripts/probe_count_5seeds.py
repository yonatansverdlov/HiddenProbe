#!/usr/bin/env python3
"""Figure-3 probe-count benchmark: HiddenProbe vs ProbeGen, 5 independent seeds.

Use the validated method-specific configurations already recorded in
scripts/{HiddenProbe,ProbeGen}/, varying ONLY the number of probes and seed.

  python scripts/probe_count_5seeds.py --dataset svhn --dry-run
  python scripts/probe_count_5seeds.py --dataset svhn --max-runs 1
  python scripts/probe_count_5seeds.py --dataset svhn

Results: checkpoints/probe_count_5seeds/<dataset>/{per_seed,summary}.csv
Test inference is timed/synchronized and the CUDA allocation peak is captured
by the existing trainer's best-validation-checkpoint evaluation, not by an
extra test-set pass.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROBES = (16, 32, 64, 128, 256)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
METHODS = ("hiddenprobe", "probegen")
DATASETS = ("fmnist_inr", "svhn", "mnist_transformer")
METRIC = {"fmnist_inr": "test_acc", "svhn": "test_tau", "mnist_transformer": "test_tau"}


def config(dataset: str, method: str, n_probes: int, seed: int,
           out_root: Path) -> tuple[list[str], Path]:
    """Immutable best-config flags mirrored from the corresponding saved runner."""
    name = f"probe_count5_{dataset}_{method}_Q{n_probes}_s{seed}"
    common = ["--n_probes", str(n_probes), "--seed", str(seed)]
    main = [sys.executable, str(ROOT / "main.py")]

    if dataset == "fmnist_inr":
        if method == "hiddenprobe":
            fixed = shlex.split("""
                --epochs 30 --batch_size 64 --d_hid 256 --mixer_n_layers 6
                --gen_type linear_2_no_acts --gen_latent_z 32 --generator_width 16
                --scheduler plateau --plateau_monitor val_acc
                --plateau_patience 3 --plateau_factor 0.2 --plateau_min_lr 1e-6
                --lr 7e-4 --weight_decay 0.0 --eval_every 500 --n_workers 0
                --per_probe_mlp mlp2 --per_probe_mlp_width 256
                --per_probe_out_dim 4 --per_probe_init standard
                --r_per_hidden 2 --rank 8 --device cuda
            """)
        else:
            fixed = shlex.split("""
                --epochs 30 --batch_size 32 --d_hid 256 --mixer_n_layers 6
                --gen_type linear_2_no_acts
                --per_probe_mlp mlp2 --per_probe_mlp_width 256
                --per_probe_out_dim 4 --per_probe_init standard
                --lr 3e-4 --eval_every 500 --n_workers 0 --device cuda
            """)
        cmd = main + [
            "--method", method, "--task", "classification", "--dataset", "fmnist",
            "--exp_name", name, "--num_seeds", "1",
        ] + common + fixed
        # The core trainer's path is fixed by its own run naming convention.
        run_dir = ROOT / "experiments" / "classification" / "fmnist" / "runs" / name
        return cmd, run_dir

    if dataset == "svhn":
        if method == "hiddenprobe":
            zoo = ROOT / "data" / "regression" / "svhn"
            run_dir = out_root / method / f"Q{n_probes}" / f"seed_{seed}"
            fixed = [
                "--zoo", "svhn_gs", "--gen_type", "deep_linear_6",
                "--models_c_in", "1", "--zoo_data_dir", str(zoo),
                "--zoo_split", str(zoo / "split.csv"),
            ] + shlex.split("""
                --probe_sharing shared --assert_canonical 1 --target_space raw
                --adapter_preset expressive --interaction_rank 96
                --mixer_hidden 384 --hidden_dim 0
                --lr 3e-4 --probe_lr 6e-4 --batch_size 32
                --rank_loss_w 0.0 --weight_decay 0.0
                --scheduler plateau --plateau_factor 0.7
                --plateau_patience 4 --plateau_min_lr 3e-5
                --epochs 150 --eval_every 500 --eval_cnn_bs 256
            """)
            cmd = main + [
                "--method", method, "--task", "regression", "--dataset", "svhn",
                "--exp_name", name, "--out_dir", str(run_dir),
            ] + common + fixed
        else:
            fixed = shlex.split("""
                --epochs 150 --batch_size 64 --d_hid 351 --mixer_n_layers 6
                --gen_type deep_linear_6 --gen_latent_z 32
                --generator_width 16 --per_probe_mlp mlp2
                --per_probe_mlp_width 256 --per_probe_out_dim 10
                --r_per_hidden 2 --scheduler plateau --plateau_monitor val_tau
                --plateau_patience 3 --plateau_factor 0.2 --plateau_min_lr 1e-6
                --lr 7e-4 --weight_decay 0.0
                --eval_every 500 --n_workers 0 --device cuda
            """)
            cmd = main + [
                "--method", method, "--task", "regression", "--dataset", "svhn",
                "--exp_name", name, "--num_seeds", "1",
            ] + common + fixed
            run_dir = ROOT / "experiments" / "regression" / "svhn" / "runs" / name
        return cmd, run_dir

    if dataset == "mnist_transformer":
        run_dir = out_root / method / f"Q{n_probes}" / f"seed_{seed}"
        data_root = ROOT / "data"
        if method == "hiddenprobe":
            fixed = shlex.split("""
                --generator g3 --n_classes 10 --ffn 384
                --readout multi --readout_arch channel_trajectory
                --cut_off 0 --pred_lr 2.5e-4 --gen_lr 5e-4
                --scheduler plateau --plateau_patience 5 --plateau_factor 0.5
                --warmup 0 --weight_decay 1e-3 --dropout 0.1
                --max_updates 40000 --eval_every 1000 --micro 32 --exact_accum
            """)
        else:
            fixed = shlex.split("""
                --generator g3 --n_classes 10 --ffn 0
                --readout multi --pma_seeds 4 --readout_arch rout
                --cut_off 0 --pred_lr 5e-4 --gen_lr 5e-4
                --scheduler cosine --plateau_patience 0 --plateau_factor 0
                --weight_decay 1e-3 --dropout 0.1 --warmup 0
                --max_updates 40000 --eval_every 1000 --micro 32
            """)
        cmd = main + [
            "transformer", "train", "--method", method, "--dataset", "mnist",
            "--data_root", str(data_root), "--runs_dir", str(run_dir),
        ] + common + fixed
        return cmd, run_dir

    raise ValueError(dataset)


def prepare(dataset: str, env: dict[str, str]) -> None:
    scripts = {
        "fmnist_inr": "classification_fmnist.sh",
        "svhn": "regression_svhn.sh",
        "mnist_transformer": "transformer_mnist.sh",
    }
    setup = ROOT / "scripts" / "setup_data" / scripts[dataset]
    subprocess.run(["bash", str(setup)], cwd=ROOT, env=env, check=True)
    if dataset == "mnist_transformer":
        subprocess.run([
            sys.executable, str(ROOT / "main.py"), "transformer", "cache",
            "--dataset", "mnist", "--seed", "0", "--cut_off", "0",
            "--data_root", str(ROOT / "data"),
        ], cwd=ROOT, env=env, check=True)


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[probe-count] Running: {shlex.join(command)}", flush=True)
    with log_path.open("w", buffering=1) as log:
        with subprocess.Popen(
            command, cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            ret = process.wait()
    if ret:
        raise RuntimeError(
            f"Training failed (exit {ret}); consult {log_path}\n"
            f"Command: {shlex.join(command)}"
        )


def stats(values: list[float]) -> tuple[float, float, float]:
    avg = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return avg, sd, sd / math.sqrt(len(values))


def export(dataset: str, probes: list[int], seeds: list[int], methods: list[str],
           out_root: Path) -> None:
    items = []
    for method in methods:
        for q in probes:
            for seed in seeds:
                _, run_dir = config(dataset, method, q, seed, out_root)
                path = run_dir / "probe_count_eval.json"
                if not path.exists():
                    continue
                with path.open() as handle:
                    row = json.load(handle)
                required = {"n_probes", "seed", "method", "test_metric_name",
                            "test_metric", "eval_time_sec", "eval_peak_gpu_mb",
                            "eval_incremental_gpu_mb", "trainable_params", "train_time_sec"}
                if not required.issubset(row):
                    raise RuntimeError(f"Incomplete benchmark record: {path}")
                if (row["method"], row["n_probes"], row["seed"]) != (method, q, seed):
                    raise RuntimeError(f"Mismatched benchmark record: {path}")
                if row["test_metric_name"] != METRIC[dataset]:
                    raise RuntimeError(f"Wrong test metric in {path}")
                items.append(row)

    out_root.mkdir(parents=True, exist_ok=True)
    if items:
        fields = [
            "dataset", "method", "n_probes", "seed", "test_metric_name", "test_metric",
            "eval_n_examples", "eval_time_sec", "eval_examples_per_sec",
            "eval_gpu_baseline_mb", "eval_peak_gpu_mb", "eval_incremental_gpu_mb",
            "train_time_sec", "trainable_params", "gpu_name",
            "torch_version", "eval_batch_size", "eval_memory_definition",
        ]
        with (out_root / "per_seed.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for item in items:
                writer.writerow(item)

    group_fields = [
        "dataset", "method", "n_probes", "n_seeds", "test_metric_name",
    ]
    measures = [
        "test_metric", "eval_time_sec", "eval_examples_per_sec",
        "eval_peak_gpu_mb", "eval_incremental_gpu_mb",
        "train_time_sec", "trainable_params",
    ]
    summary = []
    for method in methods:
        for q in probes:
            subset = [r for r in items if r["method"] == method and r["n_probes"] == q]
            # Do not imply a five-seed result from a partial, interrupted run.
            if len(subset) != len(seeds):
                continue
            row = {"dataset": dataset, "method": method, "n_probes": q,
                   "n_seeds": len(subset), "test_metric_name": METRIC[dataset]}
            for key in measures:
                v = [float(r[key]) for r in subset]
                if any(not math.isfinite(x) for x in v):
                    raise RuntimeError(f"Invalid {key} for {method}, Q={q}")
                mean, sd, sem = stats(v)
                row[key + "_mean"] = mean
                row[key + "_sd"] = sd
                row[key + "_sem"] = sem
            summary.append(row)
    names = group_fields + [f"{key}_{kind}" for key in measures for kind in ("mean", "sd", "sem")]
    with (out_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(summary)
    for item in summary:
        print(
            f"[summary] {item['method']:12s} Q={item['n_probes']:3d}: "
            f"{METRIC[dataset]}={item['test_metric_mean']:.5f} "
            f"+/- {item['test_metric_sem']:.5f} SEM; "
            f"eval={item['eval_time_sec_mean']:.2f}s; "
            f"GPU peak={item['eval_peak_gpu_mb_mean']:.1f}MiB; "
            f"GPU incremental={item['eval_incremental_gpu_mb_mean']:.1f}MiB",
            flush=True,
        )
    print(f"[probe-count] {len(items)} / {len(methods) * len(probes) * len(seeds)} seeds "
          f"complete; {len(summary)} fully aggregated groups. Files: {out_root}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--probes", nargs="+", type=int, default=list(DEFAULT_PROBES))
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--out-root", default="checkpoints/probe_count_5seeds")
    p.add_argument("--skip-setup", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-runs", type=int, default=0,
                   help="Run at most this many missing seeds (smoke test: 1); 0=all")
    p.add_argument("--eval-batch-size", type=int, default=32,
                   help="Identical test evaluation batch size for both FMNIST models")
    a = p.parse_args()
    if any(q < 1 for q in a.probes) or any(s < 0 for s in a.seeds):
        p.error("probe counts must be positive and seeds nonnegative")
    if len(set(a.probes)) != len(a.probes) or len(set(a.seeds)) != len(a.seeds):
        p.error("duplicate --probes or --seeds")
    if len(set(a.methods)) != len(a.methods):
        p.error("duplicate --methods")
    if a.dataset == "mnist_transformer" and "hiddenprobe" in a.methods:
        bad = set(a.probes) - set(DEFAULT_PROBES)
        if bad:
            p.error(f"MNIST channel-trajectory supports only {DEFAULT_PROBES}, got {sorted(bad)}")
    if a.eval_batch_size < 1:
        p.error("--eval-batch-size must be >= 1")

    out_root = (ROOT / a.out_root / a.dataset).resolve()
    env = dict(os.environ)
    env["PROBE_COUNT_PROFILE"] = "1"
    env["PROBE_COUNT_EVAL_BS"] = str(64 if a.dataset == "svhn" else a.eval_batch_size)
    env["PYTHONUNBUFFERED"] = "1"
    env["DATA_ROOT"] = str(ROOT / "data")

    todo = []
    for method in a.methods:
        for q in a.probes:
            for seed in a.seeds:
                cmd, run_dir = config(a.dataset, method, q, seed, out_root)
                profile = run_dir / "probe_count_eval.json"
                if profile.exists():
                    print(f"[probe-count] DONE {method} Q={q} seed={seed}: {profile}")
                    continue
                if (run_dir / "summary.json").exists() or (run_dir / "last.pt").exists():
                    raise RuntimeError(
                        f"{run_dir} has a completed training result but no profiled evaluation. "
                        "Refusing to overwrite it. Move that run aside and rerun, or use "
                        "a fresh --out-root (core backend uses the fixed experiments/ layout)."
                    )
                todo.append((method, q, seed, cmd, run_dir))

    if a.dry_run:
        for method, q, seed, command, run_dir in todo:
            print(f"{method} Q={q} seed={seed} -> {run_dir}\n  {shlex.join(command)}")
        print(f"[probe-count] {len(todo)} pending commands, no training started.")
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available. Run this experiment inside a CUDA/Slurm job.")
    print(f"[probe-count] Device: {torch.cuda.get_device_name()}", flush=True)
    print("[probe-count] TEST is evaluated only with the validation-selected best checkpoint. "
          "Runs execute sequentially to avoid GPU memory/timing interference.", flush=True)
    if todo and not a.skip_setup:
        prepare(a.dataset, env)

    done_now = 0
    for method, q, seed, cmd, run_dir in todo:
        if a.max_runs and done_now >= a.max_runs:
            break
        log = out_root / "logs" / f"{method}_Q{q}_s{seed}.log"
        run_command(cmd, log, env)
        if not (run_dir / "probe_count_eval.json").exists():
            raise RuntimeError(f"Trainer did not create benchmark JSON: {run_dir}")
        done_now += 1
        export(a.dataset, a.probes, a.seeds, a.methods, out_root)

    export(a.dataset, a.probes, a.seeds, a.methods, out_root)


if __name__ == "__main__":
    main()
