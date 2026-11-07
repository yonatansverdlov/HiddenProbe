#!/usr/bin/env python3
"""Forward/End-to-End/GPU-memory benchmark for Figure-3 tasks beyond FMNIST.

Datasets: SVHN CNN zoo, MNIST-Transformers (threshold 0%), CIFAR10-Wild-Park.
Methods: ProbeGen, HiddenProbe. Q: 16,32,64,128,256.

Uses the actual architectures/configurations of the experiment runners. For each
(method,Q) a NEW process builds a randomly initialized predictor, loads the
canonical TRAIN target population, warms up, then evaluates one complete pass.
There is NO training, no checkpoint restore, and no claim about accuracy.

Time definitions:
  model_compute_s: synchronized predictor forward, excludes target transfers.
  target_transfer_s: transfer of current target-network weights to GPU.
  measured_pass_wall_s: all measured passes (including transfer, cleanup).
  end_to_end_s: CPU dataset load + model setup + measured_pass_wall_s.
Warmup and Python interpreter startup are excluded from end_to_end_s.

All target networks/weights are kept on CPU between batches, so GPU peak
reflects the predictor and one target batch rather than an entire resident zoo.
This differs from the original SVHN HiddenProbe trainer's GPU-resident zoo.
Both methods use the same target split, order, batch size and precision.

Run from repository root, on an allocated GPU:
  python measurments/benchmark_other_probe_cost.py --dataset svhn
  python measurments/benchmark_other_probe_cost.py --dataset mnist_transformer
  python measurments/benchmark_other_probe_cost.py --dataset cifar10_wp
  python measurments/benchmark_other_probe_cost.py --dataset svhn --limit 32
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

DATASETS = ("svhn", "mnist_transformer", "cifar10_wp")
METHODS = ("probegen", "hiddenprobe")
DEFAULT_Q = (16, 32, 64, 128, 256)
COLS = (
    "dataset", "method", "probes", "phase", "split", "num_models", "limit",
    "batch_size", "seed", "precision", "device", "gpu", "params",
    "trainable_params", "repeats", "warmup_batches", "data_load_s",
    "model_setup_s", "target_transfer_s", "model_compute_s",
    "compute_ms_per_network", "measured_pass_wall_s", "end_to_end_s",
    "cuda_peak_allocated_mib", "cuda_peak_reserved_mib",
)
WP_DIR = ROOT / "data/regression/cifar10_wp"


def args_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    p.add_argument("--probes", type=int, nargs="+", default=list(DEFAULT_Q))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="0 = entire training split")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true", help="Replace an existing CSV")
    p.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_method", choices=METHODS, help=argparse.SUPPRESS)
    p.add_argument("--_probes", type=int, help=argparse.SUPPRESS)
    p.add_argument("--_json", type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.batch_size < 1 or a.repeats < 1 or a.warmup < 0 or a.limit < 0:
        p.error("batch-size and repeats must be positive; limit/warmup >= 0")
    if not a.probes or min(a.probes) < 1 or len(a.probes) != len(set(a.probes)):
        p.error("--probes must contain distinct positive integers")
    if len(a.methods) != len(set(a.methods)):
        p.error("--methods must not contain duplicates")
    if a.csv is None:
        a.csv = ROOT / "measurments" / f"{a.dataset}_probe_cost.csv"
    return a


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(dataset, method, q, seed):
    # Match the configurations recorded in scripts/probe_count_5seeds.py
    # (SVHN, MNIST Transformer) and the canonical CIFAR-WP runners.
    if dataset == "mnist_transformer":
        from models.transformer.system import TPConfig, LearnedSystem, fit_ffn
        is_hidden = method == "hiddenprobe"
        arch = "channel_trajectory" if is_hidden else "rout"
        ffn = 384 if is_hidden else fit_ffn(
            "mnist", "g3", 10, q, readout="multi", pma_seeds=4,
            readout_arch="rout")
        cfg = TPConfig("mnist", "g3", 10, ffn, n_probes=q, readout="multi",
                       pma_seeds=1 if is_hidden else 4, seed=seed,
                       readout_arch=arch, dropout=0.1)
        return LearnedSystem(cfg)

    in_channels = 1 if dataset == "svhn" else 3
    generator = "deep_linear_6" if dataset == "svhn" else "deep_linear_5"
    if method == "probegen":
        from models.probegen_core import ProbeGen
        svhn = dataset == "svhn"
        return ProbeGen(
            n_tokens=q, d_hidden=351 if svhn else 256,
            models_c_in=in_channels, models_c_out=10, d_out=1,
            gen_type=generator, gen_latent_z=32, generator_width=16,
            mixer_n_layers=6, include_hidden_features=False,
            per_probe_mlp="mlp2" if svhn else "none",
            per_probe_mlp_width=256 if svhn else None,
            per_probe_out_dim=10 if svhn else None,
            per_probe_init="standard", n_hidden_target_layers=0,
            r_per_hidden=2, rank=8, seed=seed,
        )

    from models.probegen_h import ProbeGenH
    svhn = dataset == "svhn"
    return ProbeGenH(
        n_out_probes=q, n_hidden_probes=q, n_classes=10,
        gen_latent_z=32, generator_width=16,
        gen_n_layers=6 if svhn else 5,
        mixer_hidden=384 if svhn else 256, mixer_n_layers=6,
        hidden_dim=128 if svhn else 64,
        z_hidden_dim=128 if svhn else 64,
        interaction_rank=96 if svhn else 32,
        fusion_hidden=256 if svhn else 128,
        fusion_out=128 if svhn else 64,
        use_hidden_statistics=False, hidden_mode="on",
        probe_sharing="shared", spatial_grid=4,
        models_c_in=in_channels, gen_type=generator,
        hidden_agg="neuron_collapse", probe_mixer="none",
    )


def load_targets(a):
    if a.dataset == "mnist_transformer":
        from models.transformer import intake as IN, teacher as T
        cache = Path(IN.cache_path(str(a.data_root), "mnist", "train", 0, 0.0))
        if not cache.is_file():
            raise FileNotFoundError(
                f"Missing MNIST-Transformer cache: {cache}\n"
                "Run: python main.py transformer cache --dataset mnist "
                "--seed 0 --cut_off 0 --data_root data")
        zoo = IN.load_zoo_cached(str(a.data_root), "mnist", "train",
                                 seed=0, cut_off=0, limit=0)
        # The training loop drops teachers whose classifier output is not C=10.
        targets = [z["params"] for z in zoo
                   if T.classifier_out_dim(z["params"]) == 10]
        if a.limit:
            targets = targets[:a.limit]
        return targets

    from data import load_cnns, load_svhn_cnns
    if a.dataset == "svhn":
        d = a.data_root / "regression/svhn"
        for fn in ("split.csv", "weights.npy", "layout.csv", "metrics.csv.gz"):
            if not (d / fn).is_file():
                raise FileNotFoundError(f"Missing SVHN zoo file: {d / fn}")
        targets, _ = load_svhn_cnns(
            "train", dev="cpu", data_dir=str(d),
            split_csv=str(d / "split.csv"), limit=a.limit)
    else:
        d = a.data_root / "regression/cifar10_wp"
        cache_dir = d / "wp_cnn_cache"
        splits = d / "splits.json"
        for f in (splits, cache_dir / "cnn_cache_train.pt"):
            if not f.is_file():
                raise FileNotFoundError(
                    f"Missing CIFAR-WP file: {f}\n"
                    "Run: bash scripts/setup_data/regression_cifar10_wp.sh")
        targets, _ = load_cnns(
            "train", limit=a.limit, dev="cpu",
            splits_path=str(splits), cnn_cache=str(cache_dir))
    for net in targets:
        net.eval().requires_grad_(False)
    return targets


def run_child(a):
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; allocate a GPU or use --device cpu")
    if a.precision == "bf16" and (device.type != "cuda" or
                                  not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 requires a CUDA GPU with BF16 support")

    seed_everything(a.seed)
    t = time.perf_counter()
    targets = load_targets(a)
    data_load_s = time.perf_counter() - t
    if not targets:
        raise ValueError("Empty training split after filtering")
    n = len(targets)

    t = time.perf_counter()
    model = build_model(a.dataset, a._method, a._probes, a.seed).float().to(device)
    model.eval()
    sync(device)
    model_setup_s = time.perf_counter() - t
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def execute(batch):
        # Always make targets GPU-resident ONLY for the current batch.
        sync(device)
        t0 = time.perf_counter()
        if a.dataset == "mnist_transformer":
            gpu_targets = [{k: v.to(device) for k, v in teacher.items()}
                           for teacher in batch]
        else:
            gpu_targets = [net.to(device) for net in batch]
        sync(device)
        transfer_s = time.perf_counter() - t0

        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(a.precision == "bf16")):
                if a.dataset == "mnist_transformer":
                    from models.transformer.acquire import predict
                    prediction = predict(model, gpu_targets)
                elif a._method == "hiddenprobe":
                    prediction = model(gpu_targets, device=str(device))
                else:
                    prediction = model(nets=gpu_targets)
        sync(device)
        compute_s = time.perf_counter() - t0
        del prediction
        if a.dataset != "mnist_transformer":
            for net in gpu_targets:
                net.to("cpu")
        del gpu_targets
        sync(device)
        return transfer_s, compute_s

    # Warm up the actual model/teacher operations, but exclude from all totals.
    first = targets[:a.batch_size]
    for _ in range(a.warmup):
        execute(first)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    transfer_s = compute_s = 0.0
    sync(device)
    measured_t0 = time.perf_counter()
    for rep in range(a.repeats):
        for start in range(0, n, a.batch_size):
            dt, ct = execute(targets[start:start + a.batch_size])
            transfer_s += dt
            compute_s += ct
        print(f"  {a.dataset} {a._method} Q={a._probes}: "
              f"pass {rep+1}/{a.repeats} finished", flush=True)
    sync(device)
    measured_s = time.perf_counter() - measured_t0

    peak_alloc = (torch.cuda.max_memory_allocated(device) / 2**20
                  if device.type == "cuda" else None)
    peak_res = (torch.cuda.max_memory_reserved(device) / 2**20
                if device.type == "cuda" else None)
    row = dict(
        dataset=a.dataset, method=a._method, probes=a._probes,
        phase="forward", split="train", num_models=n, limit=a.limit,
        batch_size=a.batch_size, seed=a.seed, precision=a.precision,
        device=str(device), gpu=(torch.cuda.get_device_name(device)
                                 if device.type == "cuda" else "CPU"),
        params=param_count, trainable_params=trainable_count,
        repeats=a.repeats, warmup_batches=a.warmup,
        data_load_s=round(data_load_s, 5),
        model_setup_s=round(model_setup_s, 5),
        target_transfer_s=round(transfer_s, 5),
        model_compute_s=round(compute_s, 5),
        compute_ms_per_network=round(1000 * compute_s / (n * a.repeats), 6),
        measured_pass_wall_s=round(measured_s, 5),
        end_to_end_s=round(data_load_s + model_setup_s + measured_s, 5),
        cuda_peak_allocated_mib=(round(peak_alloc, 2) if peak_alloc is not None else ""),
        cuda_peak_reserved_mib=(round(peak_res, 2) if peak_res is not None else ""),
    )
    a._json.write_text(json.dumps(row, indent=2), encoding="utf-8")


def run_parent(a):
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if a.csv.exists() and not a.overwrite:
        with a.csv.open(newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames != list(COLS):
                raise ValueError(f"CSV has incompatible columns: {a.csv}. "
                                 "Use --csv OTHER.csv or --overwrite.")
            for row in reader:
                must_match = {"dataset": a.dataset, "limit": str(a.limit),
                              "batch_size": str(a.batch_size),
                              "seed": str(a.seed), "precision": a.precision,
                              "device": a.device, "repeats": str(a.repeats),
                              "warmup_batches": str(a.warmup), "phase": "forward"}
                if any(row[k] != v for k, v in must_match.items()):
                    raise ValueError(
                        f"Existing {a.csv} has different settings; "
                        "use --csv OTHER.csv or --overwrite.")
                completed.add((row["method"], int(row["probes"])))
    needs_header = a.overwrite or not a.csv.exists()
    mode = "w" if needs_header else "a"
    with a.csv.open(mode, newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=COLS)
        if needs_header:
            writer.writeheader()
            fp.flush()
        for q in a.probes:
            for method in a.methods:
                if (method, q) in completed:
                    print(f"[skip] {a.dataset} {method} Q={q}: already recorded",
                          flush=True)
                    continue
                with tempfile.TemporaryDirectory(prefix="probe_cost_") as td:
                    outfile = Path(td) / "result.json"
                    cmd = [
                        sys.executable, str(Path(__file__).resolve()),
                        "--_child", "--_method", method, "--_probes", str(q),
                        "--_json", str(outfile), "--dataset", a.dataset,
                        "--batch-size", str(a.batch_size), "--limit", str(a.limit),
                        "--repeats", str(a.repeats), "--warmup", str(a.warmup),
                        "--seed", str(a.seed), "--precision", a.precision,
                        "--device", a.device, "--data-root", str(a.data_root),
                    ]
                    print(f"[run] {a.dataset} {method} Q={q}", flush=True)
                    subprocess.run(cmd, check=True)
                    row = json.loads(outfile.read_text(encoding="utf-8"))
                    writer.writerow(row)
                    fp.flush()
                    print(f"  forward={row['model_compute_s']:.3f}s "
                          f"end-to-end={row['end_to_end_s']:.3f}s "
                          f"peak={row['cuda_peak_allocated_mib']} MiB", flush=True)
    print(f"Results: {a.csv}", flush=True)


if __name__ == "__main__":
    args = args_parser()
    if args._child:
        if not (args._method and args._probes and args._json):
            raise ValueError("Missing private child arguments")
        run_child(args)
    else:
        run_parent(args)
