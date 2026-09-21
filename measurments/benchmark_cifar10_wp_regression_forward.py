#!/usr/bin/env python3
"""
Forward-pass timing benchmark:
ProbeGen vs HiddenProbe on the full CIFAR-10 Wild Park TRAIN split.

Protocol
--------
* Same target CNN objects, same order, same batches for both methods.
* Full CIFAR-10 Wild Park train split from the prebuilt CNN cache.
* Q = 128 unique target-network queries for both methods.
* Batch size = 32.
* Exactly ONE full-dataset pass by default.
* model.eval() + torch.no_grad().
* BF16 autocast for both methods.
* Measures forward pass ONLY:
    - dataset/cache loading is outside the timer
    - target-CNN construction is outside the timer
    - CPU->GPU transfer is outside the timer
    - predictor construction is outside the timer
* No separate warmup pass.
* CUDA is synchronized immediately before and after each complete pass.
* Architectures are parameter-matched as closely as possible:
    - HiddenProbe: exact CIFAR-WP runner architecture
    - ProbeGen: canonical CIFAR-WP output-only architecture with d_hidden=287
      (instead of 256) to match HiddenProbe's trainable parameter count.

Run from the repository root:
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
        description="Measure ProbeGen vs HiddenProbe CIFAR-WP forward time."
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-probes", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--cnn-cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = full TRAIN split; >0 is only for debugging.",
    )
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_probegen(n_probes: int, seed: int) -> ProbeGen:
    # CIFAR-WP ProbeGen recipe:
    # Q=128, output-only, deep_linear_5, 6-layer mixer.
    #
    # Canonical d_hidden is 256.  For this timing comparison we use 287 solely
    # to capacity-match HiddenProbe as closely as possible.
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
    # Exact architecture from:
    # scripts/HiddenProbe/regression/run_cifar10_wp_regression.sh
    #
    # adapter_preset=compact resolves to:
    # hidden_dim=64, z_hidden_dim=64, fusion_hidden=128, fusion_out=64.
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


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def timed_full_pass(
    *,
    model: torch.nn.Module,
    nets: list[torch.nn.Module],
    batch_size: int,
    device: torch.device,
    hiddenprobe: bool,
) -> float:
    model.eval()

    sync(device)
    t0 = time.perf_counter()

    with torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=(device.type == "cuda"),
    ):
        for start in range(0, len(nets), batch_size):
            batch = nets[start : start + batch_size]
            if hiddenprobe:
                _ = model(batch, device=str(device))
            else:
                _ = model(nets=batch)

    sync(device)
    return time.perf_counter() - t0


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.n_probes <= 0:
        raise ValueError("--n-probes must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    required = [
        args.cnn_cache / "cnn_cache_train.pt",
        args.splits,
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "CIFAR-WP data/cache is incomplete. Missing:\n  "
            + "\n  ".join(missing)
            + "\nRun: bash scripts/setup_data/regression_cifar10_wp.sh"
        )

    seed_all(args.seed)

    print("=" * 78)
    print("CIFAR-10 Wild Park regression forward-pass benchmark")
    print("=" * 78)
    print(f"device      : {device}")
    if device.type == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(device)}")
    print("precision   : BF16 autocast (both methods)")
    print(f"Q           : {args.n_probes}")
    print(f"batch size  : {args.batch_size}")
    print(f"repeats     : {args.repeats}")
    print("split       : TRAIN")
    print("warmup      : none")
    print()

    # ------------------------------------------------------------------
    # Load ONCE. Both methods receive the exact same target CNN objects.
    # This is measured and charged equally to both methods.
    # ------------------------------------------------------------------
    print("[1/3] Loading CIFAR-WP TRAIN split onto the device...")
    sync(device)
    data_t0 = time.perf_counter()
    nets, _ = load_cnns(
        "train",
        limit=args.limit,
        dev=str(device),
        splits_path=str(args.splits),
        cnn_cache=str(args.cnn_cache),
    )
    sync(device)
    data_load_time = time.perf_counter() - data_t0

    if not nets:
        raise RuntimeError("The CIFAR-WP train split is empty.")

    for net in nets:
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

    print(f"Loaded {len(nets):,} target CNNs in {data_load_time:.6f} s.")
    print()

    # ------------------------------------------------------------------
    # Construct the two predictors.
    # ------------------------------------------------------------------
    print("[2/3] Constructing parameter-matched ProbeGen and HiddenProbe...")
    seed_all(args.seed)
    sync(device)
    pg_build_t0 = time.perf_counter()
    probegen = make_probegen(args.n_probes, args.seed).float().to(device)
    sync(device)
    pg_build_time = time.perf_counter() - pg_build_t0

    seed_all(args.seed)
    sync(device)
    hp_build_t0 = time.perf_counter()
    hiddenprobe = make_hiddenprobe(args.n_probes).float().to(device)
    sync(device)
    hp_build_time = time.perf_counter() - hp_build_t0

    pg_total, pg_trainable = count_params(probegen)
    hp_total, hp_trainable = count_params(hiddenprobe)

    diff = pg_trainable - hp_trainable
    diff_pct = 100.0 * abs(diff) / hp_trainable

    print(f"ProbeGen    total/trainable params: {pg_total:,} / {pg_trainable:,}")
    print(f"HiddenProbe total/trainable params: {hp_total:,} / {hp_trainable:,}")
    print(f"ProbeGen    build+GPU time: {pg_build_time:.6f} s")
    print(f"HiddenProbe build+GPU time: {hp_build_time:.6f} s")
    print(
        f"Parameter difference: {diff:+,} "
        f"({diff_pct:.4f}% of HiddenProbe)"
    )

    pg_queries = args.n_probes
    hp_queries = hiddenprobe.n_target_queries()["unique_query_count"]

    print(f"ProbeGen    unique target queries/model: {pg_queries}")
    print(f"HiddenProbe unique target queries/model: {hp_queries}")
    if pg_queries != hp_queries:
        raise RuntimeError(
            f"Query-count mismatch: ProbeGen={pg_queries}, HiddenProbe={hp_queries}."
        )
    print("Query count: MATCH")
    print()

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sync(device)

    # ------------------------------------------------------------------
    # Exactly one complete pass by default.  --repeats exists only for
    # optional diagnostics; no separate warmup is performed.
    # ------------------------------------------------------------------
    print(f"[3/3] Timing {args.repeats} complete TRAIN-split pass(es)...")

    pg_times = []
    hp_times = []

    for rep in range(args.repeats):
        pg_t = timed_full_pass(
            model=probegen,
            nets=nets,
            batch_size=args.batch_size,
            device=device,
            hiddenprobe=False,
        )
        hp_t = timed_full_pass(
            model=hiddenprobe,
            nets=nets,
            batch_size=args.batch_size,
            device=device,
            hiddenprobe=True,
        )

        pg_times.append(pg_t)
        hp_times.append(hp_t)

        print(
            f"  pass {rep + 1}/{args.repeats}: "
            f"ProbeGen={pg_t:.6f}s  HiddenProbe={hp_t:.6f}s  "
            f"ratio={hp_t / pg_t:.4f}x"
        )

    pg_time = sum(pg_times) / len(pg_times)
    hp_time = sum(hp_times) / len(hp_times)
    ratio = hp_time / pg_time

    pg_forward_total = sum(pg_times)
    hp_forward_total = sum(hp_times)
    pg_end_to_end = data_load_time + pg_build_time + pg_forward_total
    hp_end_to_end = data_load_time + hp_build_time + hp_forward_total
    end_to_end_ratio = hp_end_to_end / pg_end_to_end

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"Target models               : {len(nets):,}")
    print(f"Full-dataset passes         : {args.repeats}")
    print(f"Batch size                  : {args.batch_size}")
    print("Precision                   : BF16 autocast (both)")
    print(f"Unique queries/model        : {args.n_probes}")
    print(f"ProbeGen d_hidden           : 287")
    print(f"ProbeGen trainable params   : {pg_trainable:,}")
    print(f"HiddenProbe trainable params: {hp_trainable:,}")
    print(f"Parameter difference        : {diff:+,} ({diff_pct:.4f}%)")
    print()
    print(f"Shared data load+target GPU : {data_load_time:.6f} s")
    print(f"ProbeGen build+GPU          : {pg_build_time:.6f} s")
    print(f"HiddenProbe build+GPU       : {hp_build_time:.6f} s")
    print(f"ProbeGen forward total      : {pg_forward_total:.6f} s")
    print(f"HiddenProbe forward total   : {hp_forward_total:.6f} s")
    print(f"ProbeGen END-TO-END         : {pg_end_to_end:.6f} s")
    print(f"HiddenProbe END-TO-END      : {hp_end_to_end:.6f} s")
    print(f"END-TO-END HP / PG          : {end_to_end_ratio:.4f}x")
    print()
    print(f"ProbeGen forward/pass       : {pg_time:.6f} s")
    print(f"HiddenProbe forward/pass    : {hp_time:.6f} s")
    print(f"ProbeGen ms/model           : {1000.0 * pg_time / len(nets):.6f}")
    print(f"HiddenProbe ms/model        : {1000.0 * hp_time / len(nets):.6f}")
    print(f"HiddenProbe / ProbeGen      : {ratio:.4f}x")
    print("=" * 78)


if __name__ == "__main__":
    main()
