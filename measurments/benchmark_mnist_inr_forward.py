#!/usr/bin/env python3
"""Forward-time benchmark on MNIST, FMNIST and CIFAR-10 INR train splits.

Uses original DWSNets DWS/MLP modules and original NFN modules installed at:
  third_party/benchmarks/DWSNets
  third_party/benchmarks/nfn

The upstream NFT classification config refers to NPTransformer, which is absent
from the checked upstream experiments/models.py. NFT below is *explicitly an
adapted classifier* assembled from the upstream Pointwise, Block and MlpHead
implementations; it must not be described as the original end-to-end classifier.

Each method runs in its own Python process to avoid 'experiments' and 'nn'
namespace collisions. Each sees the same INRs in the same order; no training,
no warmup, one no-grad eval pass. Checkpoint I/O/target creation and forward
time are reported separately. Use --limit 64 for an initial smoke test.

Run via:
  python measurments/benchmark_mnist_regression_forward.py --task inr --limit 64
  python measurments/benchmark_mnist_regression_forward.py --task inr --dataset mnist
Or directly:
  python measurments/benchmark_mnist_inr_forward.py --dataset fmnist
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD = ROOT / "third_party" / "benchmarks"
DWS_ROOT = THIRD / "DWSNets"
NFN_ROOT = THIRD / "nfn"
METHODS = ("probegen", "hiddenprobe", "dws", "nfn", "nft", "mlp")
DATASETS = ("mnist", "fmnist", "cifar10", "cifar10_aug")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=DATASETS, default="mnist")
    p.add_argument("--methods", default=",".join(METHODS),
                   help="Comma-separated methods (default: all six)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-probes", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--limit", type=int, default=0, help="0 = full train split")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    p.add_argument("--result-json", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.batch_size <= 0 or args.n_probes <= 0 or args.limit < 0:
        p.error("batch size and probe count must be positive; limit must be >= 0")
    args.selected = [s.strip().lower() for s in args.methods.split(",")]
    if (not args.selected or any(x not in METHODS for x in args.selected)
            or len(args.selected) != len(set(args.selected))):
        p.error("--methods must be a nonempty subset without duplicates of " + ",".join(METHODS))
    if args.worker and (not args.method or not args.result_json):
        p.error("--worker requires --method and --result-json")
    if args.csv is None:
        args.csv = ROOT / "measurments" / "results" / (args.dataset + "_inr_forward.csv")
    return args


def original_environment(method):
    env = os.environ.copy()
    if method in ("dws", "mlp"):
        base = DWS_ROOT
    elif method in ("nfn", "nft"):
        base = NFN_ROOT
    else:
        base = ROOT
    if not base.is_dir():
        raise FileNotFoundError(f"Missing original repository: {base}")
    roots = [str(base), str(ROOT)]
    if method in ("nfn", "nft"):
        roots.insert(1, str(NFN_ROOT / "experiments" / "perceiver-pytorch"))
    env["PYTHONPATH"] = os.pathsep.join(roots + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def save_csv(path, rows):
    columns = ("dataset", "method", "implementation", "targets", "batch_size",
               "n_probes", "queries_per_target", "target_params",
               "trainable_params", "all_params", "precision", "device", "seed",
               "data_s", "build_s", "forward_s", "total_s")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    os.replace(temp, path)


def parent(args):
    print("=" * 111, flush=True)
    print(f"{args.dataset.upper()} INR TRAIN | methods={','.join(args.selected)} | "
          f"batch={args.batch_size} | Q={args.n_probes} | precision={args.precision} | "
          f"limit={args.limit or 'FULL'}", flush=True)
    print("NOTE: NFT is an adapted classifier built from original NFT blocks; no original NPTransformer class.",
          flush=True)
    print("All predictors are freshly initialized. These are inference timings, NOT accuracy results.",
          flush=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="inr_benchmark_") as temp:
        for index, method in enumerate(args.selected, 1):
            dest = Path(temp) / (method + ".json")
            command = [
                sys.executable, str(Path(__file__).resolve()), "--worker",
                "--method", method, "--result-json", str(dest),
                "--dataset", args.dataset, "--data-root", str(args.data_root),
                "--batch-size", str(args.batch_size), "--n-probes", str(args.n_probes),
                "--seed", str(args.seed), "--device", args.device,
                "--precision", args.precision, "--limit", str(args.limit),
            ]
            print(f"\n[{index}/{len(args.selected)}] {method}", flush=True)
            subprocess.run(command, cwd=str(ROOT), env=original_environment(method), check=True)
            row = json.loads(dest.read_text())
            rows.append(row)
            if len({r["targets"] for r in rows}) != 1:
                raise RuntimeError("Methods saw different numbers of INRs")
            save_csv(args.csv, rows)  # Incremental results survive a later failure.
            print(f"  forward={row['forward_s']:.3f}s, total={row['total_s']:.3f}s, "
                  f"trainable={row['trainable_params']:,}", flush=True)

    print("\n" + "=" * 111)
    print(f"{'Method':<15} {'Targets':>9} {'Params':>13} {'Queries':>8} "
          f"{'Data(s)':>11} {'Build(s)':>11} {'Forward(s)':>13} {'Total(s)':>12}")
    for r in rows:
        print(f"{r['method']:<15} {r['targets']:>9,} {r['trainable_params']:>13,} "
              f"{r['queries_per_target']:>8} {r['data_s']:>11.3f} {r['build_s']:>11.3f} "
              f"{r['forward_s']:>13.3f} {r['total_s']:>12.3f}")
    print("=" * 111)
    print(f"CSV: {args.csv}")
    print("Data: checkpoint I/O, target creation where needed, and transfers.")
    print("Forward: full no-grad inference pass; GPU synchronized after every batch.")


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def import_original(method):
    if method in ("nfn", "nft"):
        # Force NFN's experiments namespace: an installed DWSNets experiments
        # package can otherwise shadow the cloned NFN experiments directory.
        import types
        sys.path.insert(0, str(NFN_ROOT))
        sys.path.insert(0, str(NFN_ROOT / "experiments" / "perceiver-pytorch"))
        pkg = types.ModuleType("experiments")
        pkg.__package__ = "experiments"
        pkg.__path__ = [str(NFN_ROOT / "experiments")]
        sys.modules["experiments"] = pkg
        from experiments.models import InvariantNFN, MlpHead, Block
        from nfn.common import WeightSpaceFeatures, network_spec_from_wsfeat
        from nfn.layers import GaussianFourierFeatureTransform, Pointwise
        return (InvariantNFN, MlpHead, Block, WeightSpaceFeatures,
                network_spec_from_wsfeat, GaussianFourierFeatureTransform, Pointwise)
    if method in ("dws", "mlp"):
        sys.path.insert(0, str(DWS_ROOT))
        from nn.models import DWSModelForClassification, MLPModelForClassification
        return DWSModelForClassification, MLPModelForClassification
    sys.path.insert(0, str(ROOT))
    from models.probegen_core import ProbeGen
    from data_probegen import INR_Network
    return ProbeGen, INR_Network


def get_paths(args):
    base = args.data_root / "classification"
    if args.dataset in ("mnist", "fmnist"):
        folder = base / (args.dataset + "_inr")
        split = folder / (args.dataset + "_splits.json")
        if not split.is_file():
            raise FileNotFoundError(f"Missing {split}; run the appropriate setup_data script")
        data = json.loads(split.read_text())["train"]
        paths = [folder / "dataset" / p for p in data["path"]]
    else:
        from data_probegen import CIFAR10INRDataset
        root = base / "cifar10_inr"
        index = CIFAR10INRDataset(
            dataset_dir=root, split="train",
            extra_aug=10 if args.dataset == "cifar10_aug" else 0,
            cache_models=False,
        )
        paths = [p for p, _label in index.samples]
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise RuntimeError("Empty INR train split")
    if not paths[0].is_file() or not paths[-1].is_file():
        raise FileNotFoundError(f"Missing INR checkpoints: {paths[0]} or {paths[-1]}")
    return paths


def load_weights(path):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "seq.0.weight" in state:
        wk = [f"seq.{i}.weight" for i in range(3)]
        bk = [f"seq.{i}.bias" for i in range(3)]
    elif "net.0.linear.weight" in state:
        wk = ["net.0.linear.weight", "net.1.linear.weight", "net.2.weight"]
        bk = ["net.0.linear.bias", "net.1.linear.bias", "net.2.bias"]
    else:
        raise ValueError(f"Unrecognized INR checkpoint {path}: {list(state)[:8]}")
    return (tuple(state[k].float().contiguous() for k in wk),
            tuple(state[k].float().contiguous() for k in bk))


def build_predictor(method, sample, libs, args, device):
    from functools import partial
    import torch
    from torch import nn
    w, b = sample
    if method in ("probegen", "hiddenprobe"):
        ProbeGen, _ = libs
        model = ProbeGen(
            n_tokens=args.n_probes, d_hidden=256,
            models_c_in=w[0].shape[1], models_c_out=w[-1].shape[0],
            d_out=10, gen_type="linear_2_no_acts", gen_latent_z=32,
            generator_width=16, mixer_n_layers=6,
            include_hidden_features=(method == "hiddenprobe"),
            per_probe_mlp="mlp2", per_probe_mlp_width=256,
            per_probe_out_dim=4, per_probe_init="standard",
            n_hidden_target_layers=2, r_per_hidden=2, rank=8, seed=args.seed,
        )
        implementation = ("HiddenProbe core, MNIST INR" if method == "hiddenprobe"
                          else "ProbeGen core, MNIST INR")
        return model.to(device), implementation
    if method in ("dws", "mlp"):
        DWS, MLP = libs
        if method == "dws":
            model = DWS(
                weight_shapes=tuple((wi.shape[1], wi.shape[0]) for wi in w),
                bias_shapes=tuple((bi.numel(),) for bi in b),
                input_features=1, hidden_dim=32, n_hidden=4,
                reduction="max", n_fc_layers=1, set_layer="sab", n_out_fc=1,
                dropout_rate=0.0, bn=True,
            )
            return model.to(device), "official DWSNets DWSModelForClassification"
        model = MLP(in_dim=sum(t.numel() for t in (w + b)),
                    hidden_dim=32, n_hidden=4, bn=True)
        return model.to(device), "official DWSNets MLPModelForClassification"
    InvariantNFN, MlpHead, Block, WSF, spec_fn, Fourier, Pointwise = libs
    sample_features = WSF(tuple(t[None, None] for t in w),
                          tuple(t[None, None] for t in b))
    spec = spec_fn(sample_features, set_all_dims=True)
    if method == "nfn":
        model = InvariantNFN(
            network_spec=spec, hchannels=[512, 512, 512],
            head_cls=partial(MlpHead, num_out=10, dropout=0.1, sigmoid=False),
            mode="HNP", feature_dropout=0.1, normalize=False,
            lnorm=None, append_stats=False,
            inp_enc_cls=partial(Fourier, mapping_size=128, scale=3),
        )
        return model.to(device), "official NFN InvariantNFN (MNIST INR config)"
    # Missing upstream NPTransformer: an honest adaptation of original NFT parts.
    model = nn.Sequential(
        Pointwise(spec, 1, 128),
        *(Block(spec, channels=128, dropout=0.1) for _ in range(4)),
        MlpHead(spec, 128, append_stats=False, num_out=10,
                pool_mode="HNP", dropout=0.0, sigmoid=False),
    )
    return model.to(device), "adapted NFT (official Pointwise + 4 Blocks + MlpHead)"


def prepare_batch(method, examples, device, libs, dataset):
    import torch
    if method in ("probegen", "hiddenprobe"):
        _, INR_Network = libs
        models = []
        for weights, biases in examples:
            net = INR_Network(
                in_features=weights[0].shape[1], n_layers=len(weights),
                hidden_features=weights[0].shape[0],
                out_features=weights[-1].shape[0],
                output_shift=0.0 if dataset.startswith("cifar10") else 0.5,
            )
            state = {}
            for i, (w, b) in enumerate(zip(weights, biases)):
                state[f"seq.{i}.weight"] = w
                state[f"seq.{i}.bias"] = b
            net.load_state_dict(state)
            net.eval().requires_grad_(False)
            models.append(net.to(device))
        return models
    if method in ("dws", "mlp"):
        # Official DWSNets data loader transposes weight dimensions and appends
        # a singleton feature dimension: (batch, input, output, 1).
        w = tuple(torch.stack([ws[i].t().unsqueeze(-1) for ws, _ in examples]).to(device)
                  for i in range(len(examples[0][0])))
        b = tuple(torch.stack([bs[i].unsqueeze(-1) for _, bs in examples]).to(device)
                  for i in range(len(examples[0][1])))
        return w, b
    WSF = libs[3]
    # NFN convention: (batch, feature_channels, output, input).
    w = tuple(torch.stack([ws[i] for ws, _ in examples]).unsqueeze(1).to(device)
              for i in range(len(examples[0][0])))
    b = tuple(torch.stack([bs[i] for _, bs in examples]).unsqueeze(1).to(device)
              for i in range(len(examples[0][1])))
    return WSF(w, b)


def worker(args):
    import torch
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for a smoke test")
    libs = import_original(args.method)  # Python import is not timed.
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    seed_everything(args.seed)
    synchronize(device)
    start_total = time.perf_counter()
    t = time.perf_counter()
    paths = get_paths(args)
    first = load_weights(paths[0])
    synchronize(device)
    data_s = time.perf_counter() - t
    target_params = sum(t.numel() for t in first[0] + first[1])

    t = time.perf_counter()
    model, implementation = build_predictor(args.method, first, libs, args, device)
    model.eval()
    synchronize(device)
    build_s = time.perf_counter() - t
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    amp = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if args.precision == "bf16" and device.type == "cuda"
           else contextlib.nullcontext)
    forward_s = 0.0
    completed = 0
    with torch.no_grad():
        for start in range(0, len(paths), args.batch_size):
            chunk = paths[start:start + args.batch_size]
            t = time.perf_counter()
            examples = [first if start == 0 and offset == 0 else load_weights(path)
                        for offset, path in enumerate(chunk)]
            batch = prepare_batch(args.method, examples, device, libs, args.dataset)
            synchronize(device)
            data_s += time.perf_counter() - t

            t = time.perf_counter()
            with amp():
                if args.method in ("probegen", "hiddenprobe"):
                    out = model(nets=batch)
                else:
                    out = model(batch)
            synchronize(device)
            forward_s += time.perf_counter() - t
            if tuple(out.shape) != (len(chunk), 10):
                raise RuntimeError(f"{args.method}: unexpected output shape {tuple(out.shape)}")
            completed += len(chunk)
            if completed % 3200 == 0 or completed == len(paths):
                print(f"  {args.method}: {completed:,}/{len(paths):,}", flush=True)
            del examples, batch, out

    synchronize(device)
    total_s = time.perf_counter() - start_total
    if completed != len(paths):
        raise RuntimeError(f"Incomplete pass: {completed}/{len(paths)}")
    result = dict(
        dataset=args.dataset, method=args.method, implementation=implementation,
        targets=completed, batch_size=args.batch_size, n_probes=args.n_probes,
        queries_per_target=args.n_probes if args.method in ("probegen", "hiddenprobe") else 0,
        target_params=target_params, trainable_params=trainable, all_params=total_params,
        precision="bf16" if args.precision == "bf16" and device.type == "cuda" else "fp32",
        device=str(device), seed=args.seed,
        data_s=data_s, build_s=build_s, forward_s=forward_s, total_s=total_s,
    )
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2))
    print(f"  COMPLETE {args.method}: total={total_s:.3f}s forward={forward_s:.3f}s", flush=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        worker(args) if args.worker else parent(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
