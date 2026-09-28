#!/usr/bin/env python
"""One-time build of the fast CNN cache (flat-tensor blob) consumed by data.load_cnns(..., cnn_cache=DIR).

Deserializes each CIFAR-WP split ONCE from the zoo zip and writes cnn_cache_<split>.pt = {flat, metas, scores}.
Subsequent training loads are ~5-6x faster (no per-CNN unzip/unpickle) and produce bit-identical CNNs.
Cache is host-local (default /dev/shm, same idea as the staged zip) -> build once per host.

Usage:
    python scripts/setup_data/build_cnn_cache.py --cache_dir data/regression/cifar10_wp/wp_cnn_cache
    python scripts/setup_data/build_cnn_cache.py --splits val --cache_dir <dir>    # one split
  The zip is taken from $PGH_WP_ZIP (default data/regression/cifar10_wp/cnn_wild_park.zip) and the split
  definition from $PGH_SPLITS (default data/regression/cifar10_wp/splits.json); see data.py.
"""
import os, io, sys, time, json, zipfile, argparse, torch
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # portable repo root
sys.path.insert(0, _REPO); os.chdir(_REPO)
from data import _stub_pickle, DEFAULT_SPLITS, DEFAULT_ZIP_SHM, DEFAULT_ZIP_NAS

def _to_plain(cfg):
    """Deep-convert an OmegaConf config to pure python so the cache loads WITHOUT
    omegaconf (portable to envs lacking it, e.g. the ant Blackwell venv). The build
    env has omegaconf (needed to unpickle the zip); the resulting cache does not."""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    # fallback: shallow dict + listify any non-plain iterable values
    out = {}
    for k, v in dict(cfg).items():
        out[k] = list(v) if (hasattr(v, "__iter__") and not isinstance(v, (str, bytes, dict))) else v
    return out


def build(split, zip_path, cache_dir, force):
    out = os.path.join(cache_dir, f"cnn_cache_{split}.pt")
    if os.path.exists(out) and not force:
        print(f"[skip] {out} exists ({os.path.getsize(out)/1e9:.1f}GB) — use --force to rebuild"); return
    lock = out + ".building"
    if os.path.exists(lock):
        print(f"[skip] {lock} present — another build in progress"); return
    open(lock, "w").close()
    try:
        zf = zipfile.ZipFile(zip_path); sp = json.load(open(DEFAULT_SPLITS))[split]
        paths, scores = sp["path"], sp["score"]; N = len(paths)
        metas = []; flats = []; off = 0; t0 = time.time()
        for i, p in enumerate(paths):
            o = torch.load(io.BytesIO(zf.read(p)), map_location="cpu", weights_only=False, pickle_module=_stub_pickle)
            sd = o["model"]; keys = list(sd.keys())
            shapes = [tuple(sd[k].shape) for k in keys]; numels = [int(sd[k].numel()) for k in keys]
            flat = torch.cat([sd[k].reshape(-1).float() for k in keys]); flats.append(flat)
            metas.append({"config": _to_plain(o["config"]), "keys": keys, "shapes": shapes, "numels": numels, "offset": off})
            off += flat.numel()
            if (i + 1) % 20000 == 0:
                print(f"  {split} {i+1}/{N} ({(i+1)/(time.time()-t0):.0f}/s)", flush=True)
        big = torch.cat(flats)
        tmp = out + ".tmp"
        torch.save({"flat": big, "metas": metas, "scores": scores}, tmp); os.replace(tmp, out)
        print(f"[built] {out}: {N} CNNs, {big.numel()/1e6:.0f}M floats, {os.path.getsize(out)/1e9:.1f}GB in {time.time()-t0:.0f}s")
    finally:
        os.path.exists(lock) and os.remove(lock)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--cache_dir", default="/dev/shm")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    zp = DEFAULT_ZIP_SHM if os.path.exists(DEFAULT_ZIP_SHM) else DEFAULT_ZIP_NAS
    print(f"zip={zp}  cache_dir={a.cache_dir}  splits={a.splits}")
    os.makedirs(a.cache_dir, exist_ok=True)
    for s in a.splits:
        build(s, zp, a.cache_dir, a.force)

if __name__ == "__main__":
    main()
