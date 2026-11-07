#!/usr/bin/env python3
"""FMNIST-INR cost benchmark: ProbeGen vs HiddenProbe, Q=16/32/64/128/256.

Measures separately:
  - model forward (or one forward+backward+optimizer step), synchronized CUDA time;
  - frozen target-INR CPU->GPU transfer time;
  - peak PyTorch GPU memory allocated and reserved during the measured pass;
  - end-to-end time (setup + data loading + transfers + computation + cleanup)
    excluding the optional warmup.

All methods use the same *frozen* Fashion-MNIST INR targets and a batch size of 32.
Each (method, Q, phase) is launched in a fresh Python process to prevent
allocator state from leaking across comparisons. This is a cost benchmark:
randomly initialized predictors, no accuracy reporting or training to convergence.

Default: entire FMNIST-INR training split, one measured eval-mode forward pass.
Pass --limit 32 for a quick smoke test, or --phase forward to match
the forward-only historical MNIST-INR timing benchmark.

Run from the repository root:
  python measurments/benchmark_fmnist_inr_probe_cost.py
  python measurments/benchmark_fmnist_inr_probe_cost.py --limit 0 --phase forward
  python measurments/benchmark_fmnist_inr_probe_cost.py --limit 32 --phase forward
"""
from __future__ import annotations

import argparse
import csv
import gc
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
import torch.nn.functional as F
from data_probegen import INRDataset
from models.probegen_core import ProbeGen

DEFAULT_DATA = ROOT / "data/classification/fmnist_inr/dataset"
DEFAULT_SPLIT = ROOT / "data/classification/fmnist_inr/fmnist_splits.json"
DEFAULT_CSV = ROOT / "measurments/fmnist_inr_probe_cost.csv"
METHODS = ("probegen", "hiddenprobe")
Q_VALUES = (16, 32, 64, 128, 256)
PHASES = ("forward", "train_step")
CSV_COLUMNS = (
    "method", "probes", "phase", "dataset", "split", "num_models", "batch_size",
    "seed", "precision", "device", "gpu", "params", "trainable_params",
    "repeats", "warmup_batches", "data_load_s", "target_transfer_s",
    "model_compute_s", "compute_ms_per_network", "model_setup_s",
    "measured_pass_wall_s", "end_to_end_s", "cuda_peak_allocated_mib",
    "cuda_peak_reserved_mib",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes", type=int, nargs="+", default=list(Q_VALUES))
    ap.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    ap.add_argument("--phase", choices=("forward", "train_step", "both"), default="forward")
    ap.add_argument("--limit", type=int, default=0,
                    help="Number of training INRs (0 = entire train split; default 0).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2,
                    help="Warmup steps on the first batch, excluded from timing.")
    ap.add_argument("--repeats", type=int, default=1, help="Measured passes through the chosen INRs.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--device", default="cuda", help="Use cuda to measure GPU memory.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_method", choices=METHODS, help=argparse.SUPPRESS)
    ap.add_argument("--_probes", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--_phase", choices=PHASES, help=argparse.SUPPRESS)
    ap.add_argument("--_json", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not args.probes or any(q <= 0 for q in args.probes):
        ap.error("all probe counts must be positive")
    if args.limit < 0 or args.batch_size <= 0 or args.repeats <= 0 or args.warmup < 0:
        ap.error("limit >= 0, batch-size > 0, repeats > 0, warmup >= 0 are required")
    return args


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_model(method: str, probes: int) -> ProbeGen:
    # Same constructor, hyperparameters and probe generator as the canonical
    # FMNIST-INR runs. The *only* method difference is hidden-feature access.
    return ProbeGen(
        n_tokens=probes,
        d_hidden=256,
        models_c_in=2,
        models_c_out=1,
        d_out=10,
        gen_type="linear_2_no_acts",
        gen_latent_z=32,
        generator_width=16,
        mixer_n_layers=6,
        include_hidden_features=(method == "hiddenprobe"),
        per_probe_mlp="mlp2",
        per_probe_mlp_width=256,
        per_probe_out_dim=4,
        per_probe_init="standard",
        n_hidden_target_layers=2,
        r_per_hidden=2,
        rank=8,
        seed=0,
    )


def run_child(args: argparse.Namespace) -> dict:
    method, q, phase = args._method, args._probes, args._phase
    if not args.data_dir.is_dir() or not args.split_file.is_file():
        raise FileNotFoundError(
            "Fashion-MNIST INR data/splits not found. Expected:\n"
            f"  {args.data_dir}\n  {args.split_file}\n"
            "Install if needed using scripts/setup_data/classification_fmnist.sh."
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable. Run on the cluster or use --device cpu for a dry run.")
    if device.type == "cuda" and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --precision fp32.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    setup_t0 = time.perf_counter()
    ds = INRDataset(dataset_dir=args.data_dir, splits_path=args.split_file, split="train")
    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    if n < 1:
        raise ValueError("The selected FMNIST-INR split contains no networks.")
    model = build_model(method, q).float().to(device)
    model.train(phase == "train_step")
    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) if phase == "train_step" else None
    use_bf16 = device.type == "cuda" and args.precision == "bf16"
    sync(device)
    model_setup_s = time.perf_counter() - setup_t0

    def load_batch(start: int) -> tuple[list, torch.Tensor, float]:
        t0 = time.perf_counter()
        samples = [ds[i] for i in range(start, min(start + args.batch_size, n))]
        nets = [net for net, _ in samples]
        labels = torch.tensor([y for _, y in samples], dtype=torch.long)
        for net in nets:
            net.eval()
            net.requires_grad_(False)
        return nets, labels, time.perf_counter() - t0

    def execute(nets: list, labels: torch.Tensor) -> tuple[float, float]:
        # Disk I/O is intentionally excluded from both transfer and compute time.
        sync(device)
        t0 = time.perf_counter()
        for net in nets:
            net.to(device)
        labels = labels.to(device)
        sync(device)
        transfer_s = time.perf_counter() - t0

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        sync(device)
        t0 = time.perf_counter()
        with torch.set_grad_enabled(phase == "train_step"):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_bf16):
                output = model(nets)
                if optimizer is not None:
                    loss = F.cross_entropy(output, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        sync(device)
        compute_s = time.perf_counter() - t0

        # Move frozen target INRs back; they must not accumulate on the GPU.
        del output
        if optimizer is not None:
            del loss
        for net in nets:
            net.to("cpu")
        del labels
        sync(device)
        return transfer_s, compute_s

    # Warmup kernels and optimizer state on a real data batch.
    warm_nets, warm_labels, _ = load_batch(0)
    for _ in range(args.warmup):
        execute(warm_nets, warm_labels)
    del warm_nets, warm_labels
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    data_s, transfer_s, compute_s = 0.0, 0.0, 0.0
    sync(device)
    measured_t0 = time.perf_counter()
    for rep in range(args.repeats):
        for start in range(0, n, args.batch_size):
            nets, labels, data_elapsed = load_batch(start)
            data_s += data_elapsed
            dt, ct = execute(nets, labels)
            transfer_s += dt
            compute_s += ct
            del nets, labels
        print(f"  {method} Q={q:3d} {phase}: repeat {rep + 1}/{args.repeats} completed",
              flush=True)

    sync(device)
    measured_pass_wall_s = time.perf_counter() - measured_t0

    peak_alloc = (torch.cuda.max_memory_allocated(device) / 2**20
                  if device.type == "cuda" else None)
    peak_res = (torch.cuda.max_memory_reserved(device) / 2**20
                if device.type == "cuda" else None)
    row = dict(
        method=method, probes=q, phase=phase, dataset="FMNIST-INR",
        split="train", num_models=n, batch_size=args.batch_size,
        seed=args.seed,
        precision=args.precision if device.type == "cuda" else "fp32",
        device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        params=param_count, trainable_params=trainable,
        repeats=args.repeats, warmup_batches=args.warmup,
        data_load_s=round(data_s, 5), target_transfer_s=round(transfer_s, 5),
        model_compute_s=round(compute_s, 5),
        compute_ms_per_network=round(1000 * compute_s / (n * args.repeats), 6),
        model_setup_s=round(model_setup_s, 5),
        measured_pass_wall_s=round(measured_pass_wall_s, 5),
        end_to_end_s=round(model_setup_s + measured_pass_wall_s, 5),
        cuda_peak_allocated_mib=round(peak_alloc, 2) if peak_alloc is not None else "",
        cuda_peak_reserved_mib=round(peak_res, 2) if peak_res is not None else "",
    )
    args._json.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def run_parent(args: argparse.Namespace) -> None:
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    phases = PHASES if args.phase == "both" else (args.phase,)
    configs = [(method, q, phase) for phase in phases
               for q in args.probes for method in args.methods]
    print(f"FMNIST-INR benchmark: {len(configs)} isolated configurations, "
          f"limit={args.limit or 'full'}, batch={args.batch_size}, "
          f"repeats={args.repeats}, phases={','.join(phases)}", flush=True)
    with args.csv.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        out.flush()
        for i, (method, q, phase) in enumerate(configs, 1):
            with tempfile.TemporaryDirectory(prefix="fmnist_inr_bench_") as td:
                result_path = Path(td) / "result.json"
                command = [
                    sys.executable, str(Path(__file__).resolve()), "--_child",
                    "--_method", method, "--_probes", str(q), "--_phase", phase,
                    "--_json", str(result_path), "--limit", str(args.limit),
                    "--batch-size", str(args.batch_size), "--warmup", str(args.warmup),
                    "--repeats", str(args.repeats), "--seed", str(args.seed),
                    "--precision", args.precision, "--device", args.device,
                    "--data-dir", str(args.data_dir), "--split-file", str(args.split_file),
                ]
                print(f"[{i}/{len(configs)}] {method} Q={q} {phase}", flush=True)
                subprocess.run(command, check=True)
                row = json.loads(result_path.read_text(encoding="utf-8"))
                writer.writerow(row)
                out.flush()  # retain completed configurations if a later one fails
                print(f"  compute={row['model_compute_s']:.3f}s "
                      f"end-to-end={row['end_to_end_s']:.3f}s "
                      f"({row['compute_ms_per_network']:.3f}ms/INR), "
                      f"peak allocated={row['cuda_peak_allocated_mib']} MiB, "
                      f"peak reserved={row['cuda_peak_reserved_mib']} MiB", flush=True)
    print(f"Results saved to {args.csv}", flush=True)


if __name__ == "__main__":
    a = parse_args()
    if a._child:
        if not (a._method and a._probes and a._phase and a._json):
            raise ValueError("Incomplete private child arguments")
        run_child(a)
    else:
        run_parent(a)
