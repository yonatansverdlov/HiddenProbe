#!/usr/bin/env python3
"""MNIST INR classification forward-time benchmark (no regression).

Methods: original DWSNets DWS/MLP, original NFN/NFT building blocks,
and this repository's ProbeGen/HiddenProbe classification implementations.

The original NFT repository contains an AutoEncoder encoder but its
experiments.models.NPTransformer classification class is missing. For NFT,
this script executes exactly the published INR2Array encoder modules and
adds a small untrained linear timing-only head. This is NOT a checkpointed
original NFT classifier; results are inference-cost measurements only.

Each method independently loads the same canonical MNIST INR TRAIN split,
builds its predictor, and performs one eval/no-grad forward over the split.
No warmup or training. Original weights are randomly initialized. Data time
covers reading INR checkpoints to CPU; forward time includes input transfer
to GPU and, for probing methods, constructing frozen target INR modules.

The historical filename is retained so existing paths keep working, but
this script exclusively benchmarks INR classification.

Smoke test:
  python measurments/benchmark_mnist_regression_forward.py --limit 32
Full train split (55,000 INRs):
  python measurments/benchmark_mnist_regression_forward.py
One method:
  python measurments/benchmark_mnist_regression_forward.py --method nfn
"""

from __future__ import annotations

import argparse
import csv
import gc
import random
import sys
import time
import traceback
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

METHODS = ("probegen", "hiddenprobe", "dws", "nfn", "nft", "mlp")
KEY_W = tuple(f"seq.{i}.weight" for i in range(3))
KEY_B = tuple(f"seq.{i}.bias" for i in range(3))
EXPECTED_W = ((32, 2), (32, 32), (1, 32))
EXPECTED_B = ((32,), (32,), (1,))
DWS_ROOT = ROOT / "third_party/benchmarks/DWSNets"
NFN_ROOT = ROOT / "third_party/benchmarks/nfn"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("all", *METHODS), default="all")
    p.add_argument("--data-dir", type=Path,
                   default=ROOT / "data/classification/mnist_inr/dataset")
    p.add_argument("--split", type=Path,
                   default=ROOT / "data/classification/mnist_inr/mnist_splits.json")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0,
                   help="0 = complete MNIST INR training split; use 32 for a smoke test")
    p.add_argument("--n-probes", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--csv", type=Path,
                   default=ROOT / "measurments/mnist_inr_forward.csv")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def add_original_repo(root, required):
    if not (root / required).is_file():
        raise FileNotFoundError(
            f"Missing original repository: {root / required}\n"
            "Clone the original DWSNets and nfn repositories into "
            "third_party/benchmarks first."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def nfn_imports():
    add_original_repo(NFN_ROOT, "experiments/models.py")
    perceiver = NFN_ROOT / "experiments/perceiver-pytorch"
    if str(perceiver) not in sys.path:
        sys.path.insert(0, str(perceiver))
    from nfn.common import WeightSpaceFeatures, network_spec_from_wsfeat
    from nfn.layers import GaussianFourierFeatureTransform, LearnedPosEmbedding
    from experiments.models import InvariantNFN, MlpHead, Block
    from experiments.inr2array import PerceiverPooling
    return (WeightSpaceFeatures, network_spec_from_wsfeat,
            GaussianFourierFeatureTransform, LearnedPosEmbedding,
            InvariantNFN, MlpHead, Block, PerceiverPooling)


def load_train_weights(args):
    """Read original checkpoints once per method without caching 55k nn.Modules."""
    from data_probegen import INRDataset

    if not args.split.is_file():
        raise FileNotFoundError(
            f"Missing MNIST INR split: {args.split}\n"
            "Run bash scripts/setup_data/classification_mnist.sh"
        )
    ds = INRDataset(
        dataset_dir=args.data_dir,
        splits_path=args.split.resolve(),
        split="train",
    )
    n = len(ds) if args.limit == 0 else min(len(ds), args.limit)
    if n < 1:
        raise ValueError("The selected MNIST INR training set is empty")
    paths = ds.dataset["path"][:n]
    first = torch.load(paths[0], map_location="cpu", weights_only=True)

    for key, shape in zip(KEY_W, EXPECTED_W):
        if tuple(first[key].shape) != shape:
            raise ValueError(f"{key} has shape {tuple(first[key].shape)} != {shape}")
    for key, shape in zip(KEY_B, EXPECTED_B):
        if tuple(first[key].shape) != shape:
            raise ValueError(f"{key} has shape {tuple(first[key].shape)} != {shape}")

    all_keys = KEY_W + KEY_B
    arrays = {
        key: torch.empty((n, *first[key].shape), dtype=first[key].dtype)
        for key in all_keys
    }
    for i, path in enumerate(paths):
        state = first if i == 0 else torch.load(
            path, map_location="cpu", weights_only=True
        )
        for key in all_keys:
            if tuple(state[key].shape) != tuple(first[key].shape):
                raise ValueError(f"Incompatible INR weights at {path}: {key}")
            arrays[key][i].copy_(state[key])
    return (
        tuple(arrays[k] for k in KEY_W),
        tuple(arrays[k] for k in KEY_B),
        n,
    )


def nfn_batch(ws, bs, start, stop, device):
    return ws(
        tuple(w[start:stop].unsqueeze(1).to(device) for w in bs[0]),
        tuple(b[start:stop].unsqueeze(1).to(device) for b in bs[1]),
    )


def dws_batch(bs, start, stop, device):
    # Original DWSNets layout: [B, in_dim, out_dim, feature_channels].
    # Original PyTorch checkpoints: [B, out_dim, in_dim].
    return (
        tuple(w[start:stop].transpose(-1, -2).unsqueeze(-1).to(device)
              for w in bs[0]),
        tuple(b[start:stop].unsqueeze(-1).to(device) for b in bs[1]),
    )


def build_nfn(spec):
    (WS, spec_fn, GFF, PosEmb, InvariantNFN, MlpHead, Block, Pool) = nfn_imports()
    return InvariantNFN(
        network_spec=spec,
        hchannels=[512, 512, 512],
        head_cls=partial(MlpHead, num_out=10, dropout=0.1),
        mode="HNP",
        feature_dropout=0.1,
        normalize=False,
        lnorm=None,
        append_stats=False,
        inp_enc_cls=partial(GFF, mapping_size=128, scale=3),
    )


def build_nft_encoder(spec):
    """Original INR2Array NFT encoder; linear head is a timing-only adapter.

    The downloaded nfn repo's nft.yaml references an unavailable NPTransformer,
    so claiming that class to be original would be misleading. The six original
    Block modules and original PerceiverPooling are used without modifications.
    """
    (WS, spec_fn, GFF, PosEmb, Inv, Head, Block, Pool) = nfn_imports()
    channels = 256
    encoder = torch.nn.Sequential(
        GFF(spec, 1, mapping_size=128, scale=3),
        PosEmb(spec, channels),
        *[
            Block(spec, channels, ff_factor=4, num_heads=4, dropout=0)
            for _ in range(6)
        ],
        Pool(
            spec, channels, n_latent=16, latent_dim=256, reduce=False,
            attn_dropout=0, ff_dropout=0, self_per_cross_attn=0,
        ),
    )
    # Encoder returns [B,16,256]; no decoder is run for classification.
    # The untrained head is explicitly not part of the original NFT encoder.
    return encoder, torch.nn.Linear(16 * 256, 10)


def build_model(method, args, data):
    ws, bs = None, data[:2]
    if method in ("nfn", "nft"):
        (WS, spec_fn, *_other) = nfn_imports()
        sample = WS(
            tuple(w[:1].unsqueeze(1) for w in bs[0]),
            tuple(b[:1].unsqueeze(1) for b in bs[1]),
        )
        spec = spec_fn(sample, set_all_dims=True)
        if method == "nfn":
            return build_nfn(spec), WS, None
        encoder, head = build_nft_encoder(spec)
        return torch.nn.ModuleDict({"encoder": encoder, "head": head}), WS, None

    if method in ("dws", "mlp"):
        add_original_repo(DWS_ROOT, "nn/models.py")
        from nn.models import DWSModelForClassification, MLPModelForClassification

        weight_shapes = tuple((w.shape[-1], w.shape[-2]) for w in bs[0])
        bias_shapes = tuple((b.shape[-1],) for b in bs[1])
        if method == "dws":
            model = DWSModelForClassification(
                weight_shapes=weight_shapes,
                bias_shapes=bias_shapes,
                input_features=1,
                hidden_dim=32,
                n_hidden=4,
                n_classes=10,
                reduction="max",
                n_fc_layers=1,
                set_layer="sab",
                n_out_fc=1,
                dropout_rate=0.0,
                bn=True,
            )
        else:
            model = MLPModelForClassification(
                in_dim=sum(int(t[0].numel()) for t in bs[0] + bs[1]),
                hidden_dim=32,
                n_hidden=4,
                n_classes=10,
                bn=True,
            )
        return model, None, None

    if method in ("probegen", "hiddenprobe"):
        from models.probegen_core import ProbeGen
        # Exactly the architecture/hyperparameters from run_mnist_inr.sh.
        # The canonical trainer forces include_hidden_features=True only
        # for HiddenProbe; all other architecture parameters are shared.
        model = ProbeGen(
            n_tokens=args.n_probes,
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
            n_hidden_target_layers=2 if method == "hiddenprobe" else 0,
            r_per_hidden=2,
            rank=8,
            seed=args.seed,
        )
        return model, None, None
    raise ValueError(method)


def build_target_batch(data, start, stop, device):
    from data_probegen import INR_Network

    weights, biases = data[:2]
    result = []
    for i in range(start, stop):
        net = INR_Network(
            in_features=2, n_layers=3, hidden_features=32, out_features=1
        ).to(device)
        state = {}
        for j in range(3):
            state[f"seq.{j}.weight"] = weights[j][i]
            state[f"seq.{j}.bias"] = biases[j][i]
        net.load_state_dict(state)
        net.eval()
        net.requires_grad_(False)
        result.append(net)
    return result


def forward_all(method, model, aux, data, args, device):
    weights, biases, n = data
    precision = args.precision == "bf16" and device.type == "cuda"
    WS = aux
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision):
            for start in range(0, n, args.batch_size):
                stop = min(n, start + args.batch_size)
                if method in ("probegen", "hiddenprobe"):
                    nets = build_target_batch(data, start, stop, device)
                    output = model(nets=nets)
                    del nets
                elif method in ("nfn", "nft"):
                    features = nfn_batch(WS, data, start, stop, device)
                    if method == "nfn":
                        output = model(features)
                    else:
                        embeddings = model["encoder"](features)
                        output = model["head"](embeddings.flatten(start_dim=1))
                else:
                    features = dws_batch(data, start, stop, device)
                    output = model(features)
                if tuple(output.shape) != (stop - start, 10):
                    raise RuntimeError(
                        f"{method}: unexpected output shape {tuple(output.shape)}"
                    )


def run_method(method, args, device):
    seed_everything(args.seed)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sync(device)

    total_start = time.perf_counter()
    t = time.perf_counter()
    data = load_train_weights(args)
    sync(device)
    data_s = time.perf_counter() - t

    t = time.perf_counter()
    model, aux, _ = build_model(method, args, data)
    model = model.float().to(device)
    model.eval()
    sync(device)
    build_s = time.perf_counter() - t
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    t = time.perf_counter()
    forward_all(method, model, aux, data, args, device)
    sync(device)
    forward_s = time.perf_counter() - t
    total_s = time.perf_counter() - total_start

    result = {
        "method": method,
        "dataset": "MNIST INR classification",
        "split": "train",
        "targets": data[2],
        "batch_size": args.batch_size,
        "seed": args.seed,
        "precision": args.precision if device.type == "cuda" else "fp32",
        "params": params,
        "trainable_params": trainable,
        "target_queries": args.n_probes if method in ("probegen", "hiddenprobe") else 0,
        "data_seconds": round(data_s, 6),
        "build_seconds": round(build_s, 6),
        "forward_seconds": round(forward_s, 6),
        "total_seconds": round(total_s, 6),
        "note": ("Original INR2Array NFT encoder + untrained linear timing head"
                 if method == "nft" else
                 "ProbeGen core with hidden features" if method == "hiddenprobe"
                 else "Original architecture, random initialization"),
    }
    del model, data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    args = arguments()
    if args.batch_size < 1 or args.n_probes < 1 or args.limit < 0:
        raise ValueError("batch-size and n-probes must be positive; limit >= 0")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU lacks BF16; use --precision fp32")

    wanted = METHODS if args.method == "all" else (args.method,)
    results, failed = [], []
    print("=" * 88, flush=True)
    print("MNIST INR CLASSIFICATION — complete TRAIN forward timing", flush=True)
    print(f"Methods: {', '.join(wanted)} | batch={args.batch_size} | "
          f"Q={args.n_probes} | limit={args.limit or 'full'} | "
          f"device={device} | precision={args.precision}", flush=True)
    print("Untrained models. No warmup. Data load repeated per method.", flush=True)
    print("Forward includes weight-to-device transfer; probe methods additionally "
          "construct and evaluate each frozen INR target.", flush=True)
    print("=" * 88, flush=True)

    for i, method in enumerate(wanted, 1):
        print(f"\n[{i}/{len(wanted)}] {method.upper()}", flush=True)
        try:
            row = run_method(method, args, device)
        except Exception:
            traceback.print_exc()
            failed.append(method)
            continue
        results.append(row)
        print(
            f"  N={row['targets']:,} params={row['trainable_params']:,} "
            f"data={row['data_seconds']:.3f}s "
            f"build={row['build_seconds']:.3f}s "
            f"forward={row['forward_seconds']:.3f}s "
            f"TOTAL={row['total_seconds']:.3f}s", flush=True
        )

    if results:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV saved: {args.csv}", flush=True)
    if failed:
        raise SystemExit(f"Failed methods: {', '.join(failed)}")
    if len(results) != len(wanted):
        raise SystemExit("Missing results")


if __name__ == "__main__":
    main()
