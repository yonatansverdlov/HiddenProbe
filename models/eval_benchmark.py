"""Optional, synchronized end-of-training evaluation benchmarks for probe-count runs.

Includes data loading/transfers and metric computation performed by the caller's
existing evaluation routine. Peak GPU memory includes live allocations at eval
start; incremental peak tracks temporary allocations above that baseline.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch


def timed_eval(fn, *, n_examples: int, device: str | torch.device):
    dev = torch.device(device)
    cuda = dev.type == "cuda" and torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize(dev)
        start_mb = torch.cuda.memory_allocated(dev) / 2**20
        torch.cuda.reset_peak_memory_stats(dev)
    else:
        start_mb = None
    start = time.perf_counter()
    result = fn()
    if cuda:
        torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated(dev) / 2**20 if cuda else None
    return result, {
        "eval_time_sec": elapsed,
        "eval_n_examples": int(n_examples),
        "eval_examples_per_sec": int(n_examples) / elapsed if elapsed > 0 else None,
        "eval_gpu_baseline_mb": start_mb,
        "eval_peak_gpu_mb": peak_mb,
        "eval_incremental_gpu_mb": max(0.0, peak_mb - start_mb) if cuda else None,
        "eval_memory_definition": (
            "torch.cuda.max_memory_allocated during full test evaluation; "
            "includes pre-existing live tensors; incremental subtracts baseline"
        ),
        "gpu_name": torch.cuda.get_device_name(dev) if cuda else None,
        "torch_version": str(torch.__version__),
    }


def save_benchmark(directory: str, *, method: str, dataset: str, seed: int,
                   n_probes: int, metric_name: str, metric: float,
                   train_time_sec: float, trainable_params: int, profile: dict):
    out = Path(directory) / "probe_count_eval.json"
    payload = {
        "method": method, "dataset": dataset, "seed": int(seed),
        "n_probes": int(n_probes), "test_metric_name": metric_name,
        "test_metric": float(metric), "train_time_sec": float(train_time_sec),
        "trainable_params": int(trainable_params),
        **profile,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out)
    print(f"[probe-count] evaluation metrics saved: {out}", flush=True)
