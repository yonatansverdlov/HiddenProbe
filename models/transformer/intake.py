"""Real SmallZoo intake (spec §2): immutable manifest + safe checkpoint loader.

Layout (verified 2026-09-11):
  <root>/{MNIST-Transformers/mnist, AG-News-Transformers/ag_news}/<run_id>_<epoch>_<acc>.pt
  epochs present: 1/50/75/100/best*. PRIMARY selection = epoch 75. Label = acc/10000 in [0,1]
  (filename encodes accuracy*100; e.g. 9835 -> 0.9835). Metadata CSVs exist for cross-check.
  AGNews checkpoints OMIT the embedding (shared w2v not shipped) -> encoder route only.

Split: seeded 70/15/15 over sorted run_ids (all states of a run stay together; we take one epoch-75
ckpt per run). Loader uses torch.load(weights_only=True) (plain tensor state-dicts) — no unpickling.
"""
from __future__ import annotations
import os, re, glob, json, hashlib, random
from typing import Dict, List
import torch

# <run_id>_<epoch>_<acc>.pt  — epoch-75 with a numeric accuracy
EP75_RE = re.compile(r"^([0-9a-z]+)_75_(\d+)\.pt$")
# dataset -> the extracted wrapper dir (glob recursively beneath it, robust to the nested subdirs)
WRAPPER = {"mnist": "mnist_transformer", "agnews": "ag_news_transformer"}


def _iter_ep75(root: str, dataset: str):
    base = os.path.join(root, WRAPPER[dataset])
    if not os.path.isdir(base):
        base = root                                  # fall back to root if wrapper absent
    for p in glob.glob(os.path.join(base, "**", "*_75_*.pt"), recursive=True):
        m = EP75_RE.match(os.path.basename(p))
        if m:
            yield m.group(1), int(m.group(2)), p


def build_manifest(root: str, dataset: str, seed: int = 0, cut_off: float = 0.0,
                   frac=(0.70, 0.15, 0.15)) -> List[Dict]:
    """Deterministic epoch-75 manifest with filter-before-split threshold protocol.

    The Transformer-NFN paper filters the epoch-75 population by absolute
    accuracy threshold first, then shuffles/splits the surviving population
    70/15/15. cut_off is expressed on the normalized [0,1] accuracy scale.
    """
    recs = {}
    for run_id, acc_raw, path in _iter_ep75(root, dataset):
        if run_id in recs:                      # exact-duplicate run_id at epoch-75 -> keep first, flag
            recs[run_id]["integrity_status"] = "duplicate_run_id"
            continue
        recs[run_id] = {
            "dataset": dataset, "run_id": run_id, "epoch": 75, "checkpoint_path": path,
            "original_label_field": "filename:acc", "original_label_value": acc_raw,
            "original_label_scale": "x100_percent", "normalized_accuracy": acc_raw / 10000.0,
            "integrity_status": "ok", "exclusion_reason": "",
        }
    if cut_off > 0:
        recs = {
            rid: r for rid, r in recs.items()
            if r["normalized_accuracy"] >= cut_off
        }

    run_ids = sorted(recs)                        # stable order after threshold filtering
    rng = random.Random(f"tp-split-{dataset}-{seed}-cut{cut_off:.6f}")
    rng.shuffle(run_ids)
    n = len(run_ids); n_tr = int(frac[0] * n); n_va = int(frac[1] * n)
    split = {}
    for i, rid in enumerate(run_ids):
        split[rid] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    out = []
    for rid in sorted(recs):
        r = dict(recs[rid]); r["split"] = split[rid]; out.append(r)
    return out


def manifest_stats(manifest: List[Dict]) -> Dict:
    from statistics import mean
    by = {"train": [], "val": [], "test": []}
    for r in manifest:
        by[r["split"]].append(r["normalized_accuracy"])
    return {s: {"n": len(v), "acc_mean": (round(mean(v), 4) if v else None),
                "acc_min": (round(min(v), 4) if v else None), "acc_max": (round(max(v), 4) if v else None)}
            for s, v in by.items()}


def write_manifest(manifest: List[Dict], path: str, overwrite: bool = False):
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing manifest {path} (pass overwrite=True)")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {"n": len(manifest), "stats": manifest_stats(manifest),
               "manifest_hash": hashlib.sha256(json.dumps(
                   [(r["run_id"], r["split"], r["normalized_accuracy"]) for r in manifest],
                   sort_keys=True).encode()).hexdigest()[:16],
               "records": manifest}
    with open(path, "w") as f:
        json.dump(payload, f, indent=0)
    return payload["manifest_hash"]


def load_target(path: str, device="cpu") -> Dict[str, torch.Tensor]:
    """Safe load -> frozen params dict keyed by the checkpoint's own state-dict names."""
    try:
        sd = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        sd = torch.load(path, map_location=device)     # fallback (documented); these are plain tensor dicts
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().to(device).float().requires_grad_(False)
    return out


def load_zoo(manifest: List[Dict], split: str, device="cpu", limit: int = 0) -> List[Dict]:
    """Load all (or `limit`) targets of a split into memory: [{id, params, label}]."""
    recs = [r for r in manifest if r["split"] == split and r["integrity_status"] == "ok"]
    if limit:
        recs = recs[:limit]
    zoo = []
    for r in recs:
        zoo.append({"id": r["run_id"], "params": load_target(r["checkpoint_path"], device),
                    "label": r["normalized_accuracy"]})
    return zoo


# ---------------- consolidated cache (avoid ~11k per-cell torch.load calls over NAS) ----------------
def _cut_tag(cut_off: float) -> str:
    pct = int(round(float(cut_off) * 100))
    return f"cut{pct:02d}"


def cache_path(root: str, dataset: str, split: str, seed: int, cut_off: float = 0.0) -> str:
    return os.path.join(root, f"tpcache_{dataset}_{split}_{_cut_tag(cut_off)}_s{seed}.pt")


def build_cache(root: str, dataset: str, seed: int = 0, cut_off: float = 0.0,
                overwrite: bool = False) -> Dict[str, object]:
    """Build one consolidated cache per threshold-specific 70/15/15 split."""
    man = build_manifest(root, dataset, seed=seed, cut_off=cut_off)
    assert man, f"no epoch-75 checkpoints under {root} for {dataset}"
    counts = {}
    for split in ("train", "val", "test"):
        path = cache_path(root, dataset, split, seed, cut_off=cut_off)
        if os.path.exists(path) and not overwrite:
            counts[split] = "exists"; continue
        zoo = load_zoo(man, split)                                     # the slow per-file load, done ONCE
        torch.save({
            "dataset": dataset,
            "split": split,
            "seed": seed,
            "cut_off": float(cut_off),
            "n": len(zoo),
            "zoo": zoo,
        }, path)
        counts[split] = len(zoo)
    return counts


def load_zoo_cached(root: str, dataset: str, split: str, seed: int = 0,
                    cut_off: float = 0.0, limit: int = 0):
    """Return the threshold-specific consolidated split if it exists."""
    path = cache_path(root, dataset, split, seed, cut_off=cut_off)
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)       # our own pickle of {id,params,label}
    zoo = d["zoo"]
    return zoo[:limit] if limit else zoo
