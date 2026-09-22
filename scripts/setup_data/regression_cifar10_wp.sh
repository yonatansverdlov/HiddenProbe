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
SPLIT_SRC="$SCRIPT_DIR/splits/cnn_park_splits.json"     # shipped with the repository
SPLIT="$TARGET/splits.json"

ARCHIVE_URL="https://zenodo.org/records/12797219/files/cnn_wild_park.zip?download=1"

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

log "Step 1: installing the shipped canonical Wild-Park split definition."
require_file "$SPLIT_SRC"
cp -f "$SPLIT_SRC" "$SPLIT"

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

log "Step 5: building the CNN cache from the zip (one pass per split; this takes a while)."
if ! ( cd "$REPO_ROOT" && PGH_WP_ZIP="$ARCHIVE" PGH_SPLITS="$SPLIT" \
        python scripts/setup_data/build_cnn_cache.py --cache_dir "$CACHE_DIR" ); then
    log "Cache build failed; assuming the archive is corrupted or incomplete."
    rm -f "$ARCHIVE" "${ARCHIVE}.part"
    log "Downloading CNN Wild Park archive again from scratch..."
    download_url "$ARCHIVE_URL" "$ARCHIVE"
    ( cd "$REPO_ROOT" && PGH_WP_ZIP="$ARCHIVE" PGH_SPLITS="$SPLIT" \
        python scripts/setup_data/build_cnn_cache.py --cache_dir "$CACHE_DIR" --force ) \
        || die "CNN Wild Park cache build failed twice."
fi

log "Step 6: running final sanity check."
cache_ready || die "$NAME setup finished, but one or more cache files are missing in $CACHE_DIR."

log "Step 7: cleaning up the downloaded archive (KEEP_ARCHIVES=1 keeps it)."
cleanup_archive "$ARCHIVE"

finish "$TARGET"
