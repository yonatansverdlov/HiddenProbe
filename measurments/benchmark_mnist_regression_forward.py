#!/usr/bin/env python3
"""
End-to-end timing benchmark:
ProbeGen vs HiddenProbe on the full MNIST-regression TRAIN split.

Each method is measured ONCE from scratch:
  data load + target-CNN construction + target-CNN GPU transfer
  + predictor construction + predictor GPU transfer
  + one full forward pass over the TRAIN split.

Both methods use:
  * the same split and target-model population
  * Q=128 unique target-network queries/model
  * batch size 32
  * model.eval() + torch.no_grad()
  * BF16 autocast
  * no warmup

Run from repository root:
    python measurments/benchmark_mnist_regression_forward.py
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from data import load_svhn_cnns
from models.probegen_core import ProbeGen
from models.probegen_h import ProbeGenH


DEFAULT_DATA_DIR = REPO_ROOT / "data" / "regression" / "mnist"
DEFAULT_SPLIT = (
    REPO_ROOT / "scripts" / "setup_data" / "splits" / "gs_splits" / "mnist_gs_auto_split.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end ProbeGen vs HiddenProbe timing on MNIST regression."
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-probes", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_probegen(n_probes: int, seed: int) -> ProbeGen:
    return ProbeGen(
        n_tokens=n_probes,
        d_hidden=300,
        models_c_in=1,
        models_c_out=10,
        d_out=1,
        gen_type="deep_linear_6",
        gen_latent_z=32,
        generator_width=16,
        mixer_n_layers=6,
        include_hidden_features=False,
        per_probe_mlp="mlp2",
        per_probe_mlp_width=256,
        per_probe_out_dim=10,
        per_probe_init="standard",
        n_hidden_target_layers=0,
        r_per_hidden=2,
        rank=8,
        seed=seed,
    )


def make_hiddenprobe(n_probes: int) -> ProbeGenH:
    return ProbeGenH(
        n_out_probes=n_probes,
        n_hidden_probes=n_probes,
        n_classes=10,
        gen_latent_z=32,
        generator_width=16,
        gen_n_layers=6,
        mixer_hidden=256,
        mixer_n_layers=6,
        hidden_dim=64,
        z_hidden_dim=64,
        interaction_rank=48,
        fusion_hidden=128,
        fusion_out=64,
        use_hidden_statistics=False,
        hidden_mode="on",
        probe_sharing="shared",
        spatial_grid=4,
        models_c_in=1,
        gen_type="deep_linear_6",
        hidden_agg="neuron_collapse",
        probe_mixer="none",
    )


def load_targets(args: argparse.Namespace, device: torch.device):
    nets, _ = load_svhn_cnns(
        "train",
        activation="relu",
        dev=str(device),
        data_dir=str(args.data_dir),
        split_csv=str(args.split),
        limit=0,
    )
    if not nets:
        raise RuntimeError("The MNIST regression train split is empty.")
    for net in nets:
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    return nets


@torch.no_grad()
def forward_full(
    model: torch.nn.Module,
    nets: list[torch.nn.Module],
    batch_size: int,
    device: torch.device,
    hiddenprobe: bool,
) -> None:
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        for start in range(0, len(nets), batch_size):
            batch = nets[start:start + batch_size]
            if hiddenprobe:
                _ = model(batch, device=str(device))
            else:
                _ = model(nets=batch)


def run_one(
    name: str,
    args: argparse.Namespace,
    device: torch.device,
    hiddenprobe: bool,
) -> dict:
    # Start timing BEFORE any per-method data/model work.
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sync(device)

    seed_all(args.seed)
    t0 = time.perf_counter()

    # 1) Data load + target construction + target GPU transfer.
    data_t0 = time.perf_counter()
    nets = load_targets(args, device)
    sync(device)
    data_time = time.perf_counter() - data_t0

    # 2) Predictor construction + GPU transfer.
    build_t0 = time.perf_counter()
    if hiddenprobe:
        model = make_hiddenprobe(args.n_probes).float().to(device)
    else:
        model = make_probegen(args.n_probes, args.seed).float().to(device)
    sync(device)
    build_time = time.perf_counter() - build_t0

    total_params, trainable_params = count_params(model)
    queries = (
        model.n_target_queries()["unique_query_count"]
        if hiddenprobe else args.n_probes
    )

    # 3) One complete forward pass.
    fwd_t0 = time.perf_counter()
    forward_full(model, nets, args.batch_size, device, hiddenprobe)
    sync(device)
    forward_time = time.perf_counter() - fwd_t0

    sync(device)
    total_time = time.perf_counter() - t0

    n_targets = len(nets)

    # Release this method completely before the next one starts.
    del model
    del nets
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sync(device)

    return {
        "name": name,
        "targets": n_targets,
        "data": data_time,
        "build": build_time,
        "forward": forward_time,
        "total": total_time,
        "params": total_params,
        "trainable": trainable_params,
        "queries": queries,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.n_probes <= 0:
        raise ValueError("--n-probes must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    required = [
        args.data_dir / "weights.npy",
        args.data_dir / "metrics.csv.gz",
        args.data_dir / "layout.csv",
        args.split,
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "MNIST regression data is incomplete. Missing:\n  "
            + "\n  ".join(missing)
            + "\nRun: bash scripts/setup_data/regression_mnist.sh"
        )

    print("=" * 78)
    print("MNIST regression END-TO-END benchmark")
    print("=" * 78)
    print(f"device      : {device}")
    if device.type == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(device)}")
    print("precision   : BF16 autocast (both)")
    print(f"Q           : {args.n_probes}")
    print(f"batch size  : {args.batch_size}")
    print("passes      : 1 per method")
    print("split       : TRAIN")
    print("warmup      : none")
    print()

    print("[1/2] ProbeGen: loading + constructing + GPU transfer + full forward...")
    pg = run_one("ProbeGen", args, device, hiddenprobe=False)
    print(f"      done in {pg['total']:.6f} s")

    print("[2/2] HiddenProbe: loading + constructing + GPU transfer + full forward...")
    hp = run_one("HiddenProbe", args, device, hiddenprobe=True)
    print(f"      done in {hp['total']:.6f} s")

    if pg["targets"] != hp["targets"]:
        raise RuntimeError(
            f"Target-count mismatch: ProbeGen={pg['targets']}, HiddenProbe={hp['targets']}"
        )
    if pg["queries"] != hp["queries"]:
        raise RuntimeError(
            f"Query-count mismatch: ProbeGen={pg['queries']}, HiddenProbe={hp['queries']}"
        )

    ratio = hp["total"] / pg["total"]

    print()
    print("=" * 78)
    print("RESULT — END TO END")
    print("=" * 78)
    print(f"Target models               : {pg['targets']:,}")
    print(f"Unique queries/model        : {pg['queries']}")
    print(f"ProbeGen trainable params   : {pg['trainable']:,}")
    print(f"HiddenProbe trainable params: {hp['trainable']:,}")
    print()
    print("ProbeGen")
    print(f"  data+targets+GPU          : {pg['data']:.6f} s")
    print(f"  predictor build+GPU       : {pg['build']:.6f} s")
    print(f"  full forward              : {pg['forward']:.6f} s")
    print(f"  TOTAL END-TO-END          : {pg['total']:.6f} s")
    print()
    print("HiddenProbe")
    print(f"  data+targets+GPU          : {hp['data']:.6f} s")
    print(f"  predictor build+GPU       : {hp['build']:.6f} s")
    print(f"  full forward              : {hp['forward']:.6f} s")
    print(f"  TOTAL END-TO-END          : {hp['total']:.6f} s")
    print()
    print(f"END-TO-END HiddenProbe / ProbeGen: {ratio:.4f}x")
    print("=" * 78)


if __name__ == "__main__":
    main()
