#!/usr/bin/env python3
"""
End-to-end timing benchmark:
ProbeGen vs HiddenProbe on the full CIFAR-10 Wild Park TRAIN split.

Each method is measured ONCE from scratch:
  cache/data load + target-CNN construction + target-CNN GPU transfer
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
    python measurments/benchmark_cifar10_wp_regression_forward.py
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

from data import load_cnns
from models.probegen_core import ProbeGen
from models.probegen_h import ProbeGenH


DEFAULT_WP_DIR = REPO_ROOT / "data" / "regression" / "cifar10_wp"
DEFAULT_CACHE = DEFAULT_WP_DIR / "wp_cnn_cache"
DEFAULT_SPLITS = REPO_ROOT / "scripts" / "setup_data" / "splits" / "cnn_park_splits.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end ProbeGen vs HiddenProbe timing on CIFAR-WP."
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-probes", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cnn-cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--limit", type=int, default=0, help="0 = full TRAIN split")
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
    # Parameter-matched CIFAR-WP output-only ProbeGen.
    return ProbeGen(
        n_tokens=n_probes,
        d_hidden=287,
        models_c_in=3,
        models_c_out=10,
        d_out=1,
        gen_type="deep_linear_5",
        gen_latent_z=32,
        generator_width=16,
        mixer_n_layers=6,
        include_hidden_features=False,
        per_probe_mlp="none",
        per_probe_mlp_width=None,
        per_probe_out_dim=None,
        per_probe_init="standard",
        n_hidden_target_layers=0,
        r_per_hidden=2,
        rank=8,
        seed=seed,
    )


def make_hiddenprobe(n_probes: int) -> ProbeGenH:
    # Exact architecture from run_cifar10_wp_regression.sh.
    return ProbeGenH(
        n_out_probes=n_probes,
        n_hidden_probes=n_probes,
        n_classes=10,
        gen_latent_z=32,
        generator_width=16,
        gen_n_layers=5,
        mixer_hidden=256,
        mixer_n_layers=6,
        hidden_dim=64,
        z_hidden_dim=64,
        interaction_rank=32,
        fusion_hidden=128,
        fusion_out=64,
        use_hidden_statistics=False,
        hidden_mode="on",
        probe_sharing="shared",
        spatial_grid=4,
        models_c_in=3,
        gen_type="deep_linear_5",
        hidden_agg="neuron_collapse",
        probe_mixer="none",
    )


def load_targets(args: argparse.Namespace, device: torch.device):
    nets, _ = load_cnns(
        "train",
        limit=args.limit,
        dev=str(device),
        splits_path=str(args.splits),
        cnn_cache=str(args.cnn_cache),
    )
    if not nets:
        raise RuntimeError("The CIFAR-WP train split is empty.")
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
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sync(device)

    seed_all(args.seed)
    t0 = time.perf_counter()

    data_t0 = time.perf_counter()
    nets = load_targets(args, device)
    sync(device)
    data_time = time.perf_counter() - data_t0

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

    fwd_t0 = time.perf_counter()
    forward_full(model, nets, args.batch_size, device, hiddenprobe)
    sync(device)
    forward_time = time.perf_counter() - fwd_t0

    sync(device)
    total_time = time.perf_counter() - t0
    n_targets = len(nets)

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
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    required = [args.cnn_cache / "cnn_cache_train.pt", args.splits]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "CIFAR-WP data/cache is incomplete. Missing:\n  "
            + "\n  ".join(missing)
            + "\nRun: bash scripts/setup_data/regression_cifar10_wp.sh"
        )

    print("=" * 78)
    print("CIFAR-10 Wild Park regression END-TO-END benchmark")
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

    diff = pg["trainable"] - hp["trainable"]
    diff_pct = 100.0 * abs(diff) / hp["trainable"]
    ratio = hp["total"] / pg["total"]

    print()
    print("=" * 78)
    print("RESULT — END TO END")
    print("=" * 78)
    print(f"Target models               : {pg['targets']:,}")
    print(f"Unique queries/model        : {pg['queries']}")
    print(f"ProbeGen trainable params   : {pg['trainable']:,}")
    print(f"HiddenProbe trainable params: {hp['trainable']:,}")
    print(f"Parameter difference        : {diff:+,} ({diff_pct:.4f}%)")
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
