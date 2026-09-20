#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NAME="Fashion-MNIST INR"
TARGET="$DATA_ROOT/classification/fmnist_inr"
DATASET_DIR="$TARGET/dataset"
ARCHIVE="$DOWNLOAD_DIR/fmnist_inrs.zip"
SPLIT="$TARGET/fmnist_splits.json"

ARCHIVE_URL="https://www.dropbox.com/sh/56pakaxe58z29mq/AABrctdu2U65jGYr2WQRzmMna/fmnist_inrs.zip?dl=1"
SPLIT_URL="https://raw.githubusercontent.com/jonkahana/ProbeGen/main/experiments/inr_classification/dataset/fmnist_splits.json"

log "============================================================"
log "Fashion-MNIST INR classification dataset setup"
log "Target: $TARGET"
log "============================================================"

require_cmd unzip
mkdir -p "$TARGET"

log "Checking whether Fashion-MNIST INR is already complete."
if [[ -s "$SPLIT" ]] \
   && find "$DATASET_DIR" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q .; then
    log "$NAME is already installed; nothing to do."
    exit 0
fi

log "Checking whether INR checkpoints are already extracted."
if find "$DATASET_DIR" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q .; then
    log "Checkpoint files already exist."
    if [[ ! -s "$SPLIT" ]]; then
        log "Only the split JSON is missing; downloading it."
        download_url "$SPLIT_URL" "$SPLIT"
    fi

    require_file "$SPLIT"
    finish "$TARGET"
    exit 0
fi

if [[ -s "$ARCHIVE" ]]; then
    log "Archive already downloaded; skipping download."
else
    log "Downloading Fashion-MNIST INR archive."
    download_url "$ARCHIVE_URL" "$ARCHIVE"
fi

PARENT="$DATA_ROOT/classification"
TMP_PREFIX=".fmnist_inr_extract."
mkdir -p "$PARENT"
cleanup_stale_extract_dirs "$PARENT" "$TMP_PREFIX"

create_tmp() {
    TMP="$(mktemp -d "$PARENT/${TMP_PREFIX}XXXXXX")"
    log "Temporary extraction directory: $TMP"
}

create_tmp
trap 'rm -rf "${TMP:-}"' EXIT

log "Extracting Fashion-MNIST INR archive."
if ! extract_zip_with_progress "$ARCHIVE" "$TMP" "Fashion-MNIST INR extraction"; then
    log "Extraction failed; treating the archive as corrupt/incomplete."
    rm -rf "$TMP"
    rm -f "$ARCHIVE" "${ARCHIVE}.part"

    log "Re-downloading archive from scratch."
    download_url "$ARCHIVE_URL" "$ARCHIVE"

    create_tmp
    if ! extract_zip_with_progress "$ARCHIVE" "$TMP" "Fashion-MNIST INR extraction"; then
        rm -f "$ARCHIVE" "${ARCHIVE}.part"
        die "Fashion-MNIST INR extraction failed twice."
    fi
fi

FIRST_PTH="$(find "$TMP" -type f -name '*.pth' -print -quit)"
[[ -n "$FIRST_PTH" ]] || die "Archive extracted, but no .pth INR checkpoints were found."
log "Found checkpoint: $FIRST_PTH"

log "Installing INR checkpoints into canonical data directory."
rm -rf "$DATASET_DIR"
mkdir -p "$DATASET_DIR"
cp -a "$TMP"/. "$DATASET_DIR"/

find "$DATASET_DIR" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q . \
    || die "No .pth INR checkpoints found after installation."

if [[ -s "$SPLIT" ]]; then
    log "Split JSON already exists."
else
    log "Downloading Fashion-MNIST split JSON."
    download_url "$SPLIT_URL" "$SPLIT"
fi

require_file "$SPLIT"
find "$DATASET_DIR" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q . \
    || die "Final sanity check failed: no INR checkpoints found."

cleanup_archive "$ARCHIVE"
finish "$TARGET"
