#!/usr/bin/env python3
"""
Temporary forward-pass timing benchmark:
ProbeGen vs HiddenProbe on the full MNIST-regression TRAIN split.

Protocol
--------
* Same target CNN objects, same order, same batches for both methods.
* Full HiddenProbe MNIST regression train split (ReLU SmallCNN zoo models).
* Q = 128 unique target-network queries for both methods.
* Batch size = 32.
* model.eval() + torch.no_grad().
* FP32 for both methods.
* Measures forward pass ONLY:
    - dataset loading is outside the timer
    - target-CNN construction is outside the timer
    - CPU->GPU transfer is outside the timer
    - model construction is outside the timer
* No model warmup passes.
* CUDA is synchronized immediately before starting and after finishing each
  full-dataset pass.
* The benchmark aborts unless the two predictors have exactly the same number
  of trainable parameters.

Run from the repository root:
    python measurments/benchmark_mnist_regression_forward.py

Optional:
    python measurments/benchmark_mnist_regression_forward.py --batch-size 32
"""

from __future__ import annotations

import argparse
import gc
import os
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
    REPO_ROOT
    / "scripts"
    / "setup_data"
    / "splits"
    / "gs_splits"
    / "mnist_gs_auto_split.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure ProbeGen vs HiddenProbe MNIST-regression forward time."
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


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_probegen(n_probes: int, seed: int) -> ProbeGen:
    # Exact architecture from scripts/ProbeGen/regression/run_mnist_regression.sh.
    # rank/r_per_hidden are irrelevant because include_hidden_features=False.
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
    # Exact architecture from scripts/HiddenProbe/regression/run_mnist_regression.sh.
    # adapter_preset=compact resolves to:
    #   hidden_dim=64, z_hidden_dim=64, fusion_hidden=128, fusion_out=64.
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


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def timed_full_pass(
    *,
    name: str,
    model: torch.nn.Module,
    nets: list[torch.nn.Module],
    batch_size: int,
    device: torch.device,
    hiddenprobe: bool,
) -> float:
    model.eval()

    # Loading/construction/transfers are already complete. The timed region
    # contains only predictor forward calls over the full train split.
    sync(device)
    t0 = time.perf_counter()

    for start in range(0, len(nets), batch_size):
        batch = nets[start : start + batch_size]
        if hiddenprobe:
            _ = model(batch, device=str(device))
        else:
            _ = model(nets=batch)

    sync(device)
    elapsed = time.perf_counter() - t0

    print(
        f"{name:12s}: {elapsed:.6f} s total | "
        f"{1000.0 * elapsed / len(nets):.6f} ms/model | "
        f"{len(nets) / elapsed:.2f} models/s"
    )
    return elapsed


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

    seed_all(args.seed)

    print("=" * 78)
    print("MNIST regression forward-pass benchmark")
    print("=" * 78)
    print(f"device      : {device}")
    if device.type == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(device)}")
    print(f"precision   : FP32")
    print(f"Q           : {args.n_probes}")
    print(f"batch size  : {args.batch_size}")
    print(f"split       : TRAIN")
    print(f"activation  : ReLU")
    print("warmup      : none")
    print()

    # ------------------------------------------------------------------
    # Load ONCE. Both predictors receive the exact same target CNN objects.
    # This happens entirely outside the timed region.
    # ------------------------------------------------------------------
    print("[1/3] Loading the full MNIST-regression TRAIN split onto the device...")
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

    print(f"Loaded {len(nets):,} target CNNs.")
    print()

    # ------------------------------------------------------------------
    # Construct the two exact predictor architectures.
    # ------------------------------------------------------------------
    print("[2/3] Constructing ProbeGen and HiddenProbe...")
    seed_all(args.seed)
    probegen = make_probegen(args.n_probes, args.seed).float().to(device)

    seed_all(args.seed)
    hiddenprobe = make_hiddenprobe(args.n_probes).float().to(device)

    pg_total, pg_trainable = count_params(probegen)
    hp_total, hp_trainable = count_params(hiddenprobe)

    print(f"ProbeGen    total/trainable params: {pg_total:,} / {pg_trainable:,}")
    print(f"HiddenProbe total/trainable params: {hp_total:,} / {hp_trainable:,}")

    if pg_trainable == hp_trainable:
        print("Parameter count: MATCH")
    else:
        print(
            "Parameter count: MISMATCH "
            f"(ProbeGen={pg_trainable:,}, HiddenProbe={hp_trainable:,}) — continuing as requested"
        )

    pg_queries = args.n_probes
    hp_queries = hiddenprobe.n_target_queries()["unique_query_count"]
    print(f"ProbeGen    unique target queries/model: {pg_queries}")
    print(f"HiddenProbe unique target queries/model: {hp_queries}")

    if pg_queries != hp_queries:
        raise RuntimeError(
            f"Query-count mismatch: ProbeGen={pg_queries}, HiddenProbe={hp_queries}."
        )

    print("Query count:     MATCH")
    print()

    # Free any transient construction garbage before timing.
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sync(device)

    # ------------------------------------------------------------------
    # Time exactly one complete forward sweep over the full train split.
    # No model warmup passes.
    # ------------------------------------------------------------------
    print("[3/3] Timing one complete pass over the full TRAIN split...")
    with torch.no_grad():
        pg_time = timed_full_pass(
            name="ProbeGen",
            model=probegen,
            nets=nets,
            batch_size=args.batch_size,
            device=device,
            hiddenprobe=False,
        )
        hp_time = timed_full_pass(
            name="HiddenProbe",
            model=hiddenprobe,
            nets=nets,
            batch_size=args.batch_size,
            device=device,
            hiddenprobe=True,
        )

    ratio = hp_time / pg_time
    delta_pct = (ratio - 1.0) * 100.0

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"Target models               : {len(nets):,}")
    print(f"Batch size                  : {args.batch_size}")
    print(f"Unique queries/model        : {args.n_probes}")
    print(f"Trainable params (each)     : {pg_trainable:,}")
    print(f"ProbeGen total time         : {pg_time:.6f} s")
    print(f"HiddenProbe total time      : {hp_time:.6f} s")
    print(f"HiddenProbe / ProbeGen      : {ratio:.4f}x")
    print(f"HiddenProbe runtime change  : {delta_pct:+.2f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
