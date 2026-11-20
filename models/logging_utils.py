"""Shared terminal logging for all HiddenProbe training backends."""

from __future__ import annotations

import statistics
import json
import os
import time
from pathlib import Path
import torch


SEP = "=" * 80


def format_duration(seconds):
    if seconds is None:
        return "n/a"
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def print_run_config(*, method, task, dataset, seed, experiment,
                     train_size, val_size, test_size,
                     probes, parameters, trainable, device):
    print(SEP)
    print("RUN CONFIG")
    print(SEP)
    print(f"Method:        {method}")
    print(f"Task:          {task}")
    print(f"Dataset:       {dataset}")
    print(f"Seed:          {seed}")
    print(f"Experiment:    {experiment}")
    print()
    print("Data:")
    print(f"  Train:       {train_size:,}")
    print(f"  Val:         {val_size:,}")
    print(f"  Test:        {test_size:,}")
    print()
    print("Model:")
    print(f"  Probes:      {probes:,}")
    print(f"  Parameters:  {parameters:,}")
    print(f"  Trainable:   {trainable:,}")
    print()
    print(f"Device:        {device}")
    print(SEP, flush=True)


def print_eval(*, task, step, epoch, train_loss, val_value, test_value,
               elapsed, remaining, new_best=False):
    if task == "classification":
        val_name, test_name = "val_acc", "test_acc"
    else:
        val_name, test_name = "val_tau", "test_tau"

    test_text = "n/a" if test_value is None else f"{float(test_value):.4f}"
    suffix = " | NEW_BEST" if new_best else ""
    print(
        f"EVAL | step={int(step)} | epoch={int(epoch)} | "
        f"train_loss={float(train_loss):.4f} | "
        f"{val_name}={float(val_value):.4f} | "
        f"{test_name}={test_text} | "
        f"elapsed={format_duration(elapsed)} | "
        f"remaining={format_duration(remaining)}{suffix}",
        flush=True,
    )


def print_seed_result(*, task, seed, best_epoch, best_step, val_value, test_value):
    if task == "classification":
        val_label, test_label = "Val accuracy", "Test accuracy"
    else:
        val_label, test_label = "Val tau", "Test tau"

    print(SEP)
    print("SEED RESULT")
    print(SEP)
    print(f"Seed:           {seed}")
    print(f"Best epoch:     {best_epoch}")
    print(f"Best step:      {best_step}")
    print(f"{val_label + ':':15s} {float(val_value):.4f}")
    print(f"{test_label + ':':15s} {float(test_value):.4f}")
    print(SEP, flush=True)


def print_final_summary(*, method, task, dataset, values):
    values = [float(v) for v in values]
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    metric = "Test accuracy" if task == "classification" else "Test tau"

    print(SEP)
    print("FINAL SUMMARY")
    print(SEP)
    print(f"Method:          {method}")
    print(f"Task:            {task}")
    print(f"Dataset:         {dataset}")
    print(f"Seeds:           {len(values)}")
    print()
    print(f"{metric + ':':17s}{mean:.4f} ± {std:.4f}")
    print("Per seed:        " + ", ".join(f"{v:.4f}" for v in values))
    print(SEP, flush=True)


# Optional evaluation timing and GPU-memory measurements.
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
