#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# CNN Wild Park is read DIRECTLY from the Zenodo zip (never extracted). Training uses a flat-tensor
# cache (cnn_cache_<split>.pt) built once from the zip; after the cache exists the zip is no longer needed.
NAME="CIFAR10-WP"
TARGET="$DATA_ROOT/regression/cifar10_wp"
ARCHIVE="$TARGET/cnn_wild_park.zip"
CACHE_DIR="${PGH_WP_CACHE:-$TARGET/wp_cnn_cache}"
SPLIT="$TARGET/splits.json"

ARCHIVE_URL="https://zenodo.org/records/12797219/files/cnn_wild_park.zip?download=1"
SPLIT_URL="https://raw.githubusercontent.com/jonkahana/ProbeGen/main/experiments/cnn_generalization/dataset/cnn_park_splits.json"

log "============================================================"
log "CIFAR10 Wild Park regression dataset setup"
log "Target directory: $TARGET"
log "Archive path:     $ARCHIVE"
log "Cache directory:  $CACHE_DIR"
log "============================================================"

require_cmd python
mkdir -p "$TARGET" "$CACHE_DIR"

cache_ready() {
    local s
    for s in train val test; do
        [[ -s "$CACHE_DIR/cnn_cache_$s.pt" ]] || return 1
    done
    return 0
}

log "Step 1: downloading the canonical Wild-Park split definition if needed."
if [[ -s "$SPLIT" ]]; then
    log "Canonical split already exists: $SPLIT"
else
    download_url "$SPLIT_URL" "$SPLIT"
fi
require_file "$SPLIT"

log "Step 2: checking whether the CNN cache is already complete."
if cache_ready; then
    log "$NAME cache already exists at: $CACHE_DIR"
    log "Nothing else to do."
    finish "$TARGET"
    exit 0
fi

log "Step 3: checking for the downloaded archive."
if [[ -s "$ARCHIVE" ]]; then
    log "Existing archive found: $ARCHIVE"
else
    log "Downloading CNN Wild Park archive (very large)..."
    download_url "$ARCHIVE_URL" "$ARCHIVE"
fi

log "Step 4: checking Python dependencies for the cache build (torch + omegaconf, needed to unpickle the zip)."
python -c 'import torch, omegaconf' 2>/dev/null \
    || die "torch and omegaconf must be importable in the active Python environment (pip install omegaconf)."

build_cache() {
    local force="${1:-0}"
    ( cd "$REPO_ROOT" && python - "$ARCHIVE" "$SPLIT" "$CACHE_DIR" "$force" <<'PY'
import io
import json
import os
import sys
import time
import zipfile

import torch

from data import _stub_pickle

zip_path, split_path, cache_dir, force_arg = sys.argv[1:]
force = force_arg == "1"

def to_plain(cfg):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass

    out = {}
    for key, value in dict(cfg).items():
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
            value = list(value)
        out[key] = value
    return out

with open(split_path, "r") as f:
    split_def = json.load(f)

os.makedirs(cache_dir, exist_ok=True)

with zipfile.ZipFile(zip_path) as zf:
    for split in ("train", "val", "test"):
        out = os.path.join(cache_dir, f"cnn_cache_{split}.pt")
        if os.path.exists(out) and not force:
            print(f"[skip] {out} already exists")
            continue

        lock = out + ".building"
        if os.path.exists(lock):
            raise RuntimeError(f"Cache build lock already exists: {lock}")

        open(lock, "w").close()
        try:
            paths = split_def[split]["path"]
            scores = split_def[split]["score"]
            n_models = len(paths)
            metas = []
            flats = []
            offset = 0
            start = time.time()

            for i, path in enumerate(paths):
                obj = torch.load(
                    io.BytesIO(zf.read(path)),
                    map_location="cpu",
                    weights_only=False,
                    pickle_module=_stub_pickle,
                )
                state_dict = obj["model"]
                keys = list(state_dict.keys())
                shapes = [tuple(state_dict[key].shape) for key in keys]
                numels = [int(state_dict[key].numel()) for key in keys]
                flat = torch.cat(
                    [state_dict[key].reshape(-1).float() for key in keys]
                )
                flats.append(flat)
                metas.append(
                    {
                        "config": to_plain(obj["config"]),
                        "keys": keys,
                        "shapes": shapes,
                        "numels": numels,
                        "offset": offset,
                    }
                )
                offset += flat.numel()

                if (i + 1) % 20000 == 0:
                    rate = (i + 1) / max(time.time() - start, 1e-9)
                    print(
                        f"  {split} {i + 1}/{n_models} ({rate:.0f}/s)",
                        flush=True,
                    )

            big = torch.cat(flats)
            tmp = out + ".tmp"
            torch.save(
                {"flat": big, "metas": metas, "scores": scores},
                tmp,
            )
            os.replace(tmp, out)
            print(
                f"[built] {out}: {n_models} CNNs, "
                f"{big.numel() / 1e6:.0f}M floats, "
                f"{os.path.getsize(out) / 1e9:.1f}GB "
                f"in {time.time() - start:.0f}s"
            )
        finally:
            if os.path.exists(lock):
                os.remove(lock)
PY
    )
}

log "Step 5: building the CNN cache from the zip (one pass per split; this takes a while)."
if ! build_cache 0; then
    log "Cache build failed; assuming the archive is corrupted or incomplete."
    rm -f "$ARCHIVE" "${ARCHIVE}.part"
    log "Downloading CNN Wild Park archive again from scratch..."
    download_url "$ARCHIVE_URL" "$ARCHIVE"
    build_cache 1 || die "CNN Wild Park cache build failed twice."
fi

log "Step 6: running final sanity check."
cache_ready || die "$NAME setup finished, but one or more cache files are missing in $CACHE_DIR."

log "Step 7: cleaning up the downloaded archive (KEEP_ARCHIVES=1 keeps it)."
cleanup_archive "$ARCHIVE"

finish "$TARGET"
