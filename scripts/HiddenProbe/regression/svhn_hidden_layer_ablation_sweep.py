#!/usr/bin/env python3
"""SVHN CNN-layer ablation: original 24-config SVHN probe-LR sweep + five final seeds."""
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# Original SVHN sweep: 3 probe LRs x 4 Plateau factors x 2 patiences.
# The readout learning rate is fixed at 3e-4; DO NOT sweep hidden_lr.
PROBE_LRS = ("1e-4", "3e-4", "6e-4")
PATIENCES = (4, 5)
FACTORS = ("0.2", "0.3", "0.5", "0.9")


def candidates():
    return [dict(probe_lr=lr, scheduler="plateau", patience=p, factor=f)
            for lr, p, f in itertools.product(PROBE_LRS, PATIENCES, FACTORS)]


def tag(c):
    return f'probeLR{c["probe_lr"]}_plateau_p{c["patience"]}_f{c["factor"]}'


def read_summary(directory, layers, seed, final):
    try:
        d = json.loads((directory / "summary.json").read_text())
        if d["seed"] != seed or d.get("hidden_layer_ids") != list(layers):
            return None
        val = float(d["best_val_tau"])
        if not math.isfinite(val):
            return None
        record = {"best_val_tau": val}
        if final:
            test = float(d["final_test_tau"])
            if not math.isfinite(test):
                return None
            record["final_test_tau"] = test
        elif not d.get("skip_test_eval"):
            # Hyperparameter search must never use test predictions.
            return None
        return record
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def csv_write(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def command(args, layers, c, epochs, seed, name, directory, final):
    zoo = Path(os.getenv("ZOO_DIR", str(Path(os.getenv("DATA_ROOT", str(ROOT / "data")))
                                      / "regression" / "svhn"))).resolve()
    cmd = [
        args.python, os.getenv("MAIN_PY", "main.py"),
        "--method", "hiddenprobe", "--task", "regression", "--dataset", "svhn",
        "--hidden_layer_ids", *map(str, layers),
        "--gen_type", "deep_linear_6", "--models_c_in", "1",
        "--zoo_data_dir", str(zoo), "--zoo_split", str(zoo / "split.csv"),
        "--probe_sharing", "shared", "--assert_canonical", "1",
        "--target_space", "raw", "--adapter_preset", "expressive",
        "--interaction_rank", "96", "--mixer_hidden", "384", "--hidden_dim", "0",
        "--n_probes", str(args.n_probes), "--lr", "3e-4", "--probe_lr", c["probe_lr"],
        "--batch_size", "32", "--rank_loss_w", "0.0", "--weight_decay", "0.0",
        "--scheduler", c["scheduler"], "--plateau_min_lr", "3e-5",
        "--epochs", str(epochs), "--eval_every", "500", "--eval_cnn_bs", "256",
        "--seed", str(seed), "--exp_name", name, "--out_dir", str(directory),
        "--skip_test_eval", "0" if final else "1",
    ]
    if c["scheduler"] == "plateau":
        cmd.extend(["--plateau_factor", c["factor"],
                    "--plateau_patience", str(c["patience"])])
    return cmd


def run_or_reuse(cmd, directory, layers, seed, final):
    record = read_summary(directory, layers, seed, final)
    if record is not None:
        print(f"[REUSE] {directory}", flush=True)
        return record
    if directory.exists():
        shutil.rmtree(directory)  # interrupted or mismatched run: restart clean
    subprocess.run(cmd, cwd=ROOT, check=True)
    record = read_summary(directory, layers, seed, final)
    if record is None:
        raise RuntimeError(f"Missing or invalid training summary: {directory}")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--n-probes", type=int, default=int(os.getenv("N_PROBES", "128")))
    p.add_argument("--sweep-epochs", type=int, default=int(os.getenv("SWEEP_EPOCHS", "20")))
    p.add_argument("--final-epochs", type=int, default=int(os.getenv("FINAL_EPOCHS", "150")))
    p.add_argument("--num-seeds", type=int, default=int(os.getenv("NUM_SEEDS", "5")))
    p.add_argument("--python", default=os.getenv("PYTHON_BIN", sys.executable))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    layers = tuple(args.layers)
    if (len(layers) not in (1, 2)
            or layers != tuple(sorted(set(layers)))
            or any(i not in (0, 1, 2) for i in layers)):
        p.error("Choose one or two distinct ascending layer IDs in {0,1,2}")
    if min(args.n_probes, args.sweep_epochs, args.final_epochs, args.num_seeds) < 1:
        p.error("All counts must be positive")
    mask = "".join(map(str, layers))
    out = ROOT / os.getenv("OUT_DIR", "checkpoints") / "hidden_layer_ablation" / "svhn" / f"layers_{mask}"
    sweep = out / f"sweep_{args.sweep_epochs}epochs"
    final = out / f"final_{args.final_epochs}epochs"
    grid = candidates()
    print(f"[SVHN] layers={layers} probes={args.n_probes} "
          f"configurations={len(grid)} sweep={args.sweep_epochs} epochs "
          f"final={args.final_epochs} epochs x {args.num_seeds} seeds", flush=True)
    if not args.dry_run:
        zoo = Path(os.getenv("ZOO_DIR", str(Path(os.getenv("DATA_ROOT", str(ROOT / "data")))
                                          / "regression" / "svhn"))).resolve()
        if not (zoo / "split.csv").is_file():
            raise FileNotFoundError(f"Missing official SVHN split: {zoo / 'split.csv'}")

    rows = []
    for i, c in enumerate(grid, start=1):
        name = f"hiddenprobe_svhn_L{mask}_SWEEP{args.sweep_epochs}_{tag(c)}_s0"
        run_dir = sweep / "runs" / name
        cmd = command(args, layers, c, args.sweep_epochs, 0, name, run_dir, False)
        print(f"[SWEEP {i}/{len(grid)}] {tag(c)}", flush=True)
        if args.dry_run:
            print(shlex.join(cmd), flush=True)
            continue
        rows.append({**c, "tag": tag(c), "seed": 0,
                     **run_or_reuse(cmd, run_dir, layers, 0, False),
                     "run_dir": str(run_dir)})
    if args.dry_run:
        print(f"[DRY RUN] {len(grid)} sweep jobs, then {args.num_seeds} final jobs")
        return
    rows.sort(key=lambda r: r["best_val_tau"], reverse=True)
    csv_write(sweep / "sweep_results.csv", rows,
              ["probe_lr", "scheduler", "patience", "factor", "tag",
               "seed", "best_val_tau", "run_dir"])
    win = rows[0]
    c = {k: win[k] for k in ("probe_lr", "scheduler", "patience", "factor")}
    (sweep / "selected_config.json").write_text(json.dumps({
        "layers": list(layers), "selected_by": "best_val_tau", "config": c,
        "best_val_tau": win["best_val_tau"]}, indent=2) + "\n")
    print(f"[WINNER] {c}: validation tau {win['best_val_tau']:.6f}", flush=True)

    final_rows = []
    for seed in range(args.num_seeds):
        name = f"hiddenprobe_svhn_L{mask}_FINAL{args.final_epochs}_{tag(c)}_s{seed}"
        run_dir = final / "runs" / name
        cmd = command(args, layers, c, args.final_epochs, seed, name, run_dir, True)
        print(f"[FINAL {seed + 1}/{args.num_seeds}] seed={seed}", flush=True)
        final_rows.append({"seed": seed,
                           **run_or_reuse(cmd, run_dir, layers, seed, True),
                           "run_dir": str(run_dir)})
    csv_write(final / "five_seeds.csv", final_rows,
              ["seed", "best_val_tau", "final_test_tau", "run_dir"])
    scores = [r["final_test_tau"] for r in final_rows]
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    result = {"dataset": "svhn", "layers": list(layers), "probes": args.n_probes,
              "selected_by": "best_val_tau", "selected_config": c,
              "sweep_epochs": args.sweep_epochs, "final_epochs": args.final_epochs,
              "num_seeds": len(scores), "test_tau_mean": statistics.mean(scores),
              "test_tau_std": std, "test_tau_sem": std / math.sqrt(len(scores))}
    (final / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[DONE] layers={layers}: test tau {result['test_tau_mean']:.6f} "
          f"+/- {result['test_tau_sem']:.6f} SEM", flush=True)


if __name__ == "__main__":
    main()
