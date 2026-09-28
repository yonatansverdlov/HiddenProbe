#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NAME="MNIST INR"
TARGET="$DATA_ROOT/classification/mnist_inr"
DATASET_DIR="$TARGET/dataset"
ARCHIVE="$DOWNLOAD_DIR/mnist-inrs-only.zip"
SPLIT="$TARGET/mnist_splits.json"

ARCHIVE_URL="https://www.dropbox.com/scl/fo/2akm78f7ot4o52o1mrtun/ADLLU8zOj73tswlhhCR_yF8/mnist-inrs.zip?rlkey=4oj9ao6om06tgmfabyctzu2n2&e=1&dl=1"
SPLIT_URL="https://raw.githubusercontent.com/jonkahana/ProbeGen/main/experiments/inr_classification/dataset/mnist_splits.json"

log "============================================================"
log "MNIST INR classification dataset setup"
log "Target: $TARGET"
log "============================================================"

require_cmd unzip
mkdir -p "$TARGET"

log "Checking whether MNIST INR is already complete."
if [[ -s "$SPLIT" ]] && [[ -d "$DATASET_DIR/mnist-inrs" ]]; then
    log "$NAME is already installed; nothing to do."
    exit 0
fi

if [[ -d "$DATASET_DIR" ]]; then
    log "Existing MNIST INR directory is incomplete; replacing it."
    rm -rf "$DATASET_DIR"
fi

if [[ -s "$ARCHIVE" ]]; then
    log "Archive already downloaded; skipping download."
else
    log "Downloading MNIST INR archive."
    download_url "$ARCHIVE_URL" "$ARCHIVE"
fi

PARENT="$DATA_ROOT/classification"
TMP_PREFIX=".mnist_inr_extract."
mkdir -p "$PARENT"
cleanup_stale_extract_dirs "$PARENT" "$TMP_PREFIX"

create_tmp() {
    TMP="$(mktemp -d "$PARENT/${TMP_PREFIX}XXXXXX")"
    log "Temporary extraction directory: $TMP"
}

create_tmp
trap 'rm -rf "${TMP:-}"' EXIT

log "Extracting MNIST INR checkpoints only."
if ! unzip -q "$ARCHIVE" 'mnist-inrs/*/checkpoints/model_final.pth' -d "$TMP"; then
    log "Extraction failed; treating the archive as corrupt/incomplete."
    rm -rf "$TMP"
    rm -f "$ARCHIVE" "${ARCHIVE}.part"

    log "Re-downloading archive from scratch."
    download_url "$ARCHIVE_URL" "$ARCHIVE"

    create_tmp
    if ! unzip -q "$ARCHIVE" 'mnist-inrs/*/checkpoints/model_final.pth' -d "$TMP"; then
        rm -f "$ARCHIVE" "${ARCHIVE}.part"
        die "MNIST INR extraction failed twice."
    fi
fi

MNIST_ROOT="$(find "$TMP" -type d -name 'mnist-inrs' -print -quit)"
[[ -n "$MNIST_ROOT" ]] || die "Archive does not contain the expected mnist-inrs directory."

log "Installing only mnist-inrs into canonical data directory."
rm -rf "$DATASET_DIR"
mkdir -p "$DATASET_DIR"
mv "$MNIST_ROOT" "$DATASET_DIR/"

[[ -d "$DATASET_DIR/mnist-inrs" ]] \
    || die "Installed MNIST INR tree is missing mnist-inrs."

if [[ -s "$SPLIT" ]]; then
    log "Split JSON already exists."
else
    log "Downloading MNIST split JSON."
    download_url "$SPLIT_URL" "$SPLIT"
fi

require_file "$SPLIT"
[[ -d "$DATASET_DIR/mnist-inrs" ]] \
    || die "Final sanity check failed: mnist-inrs directory is missing."

cleanup_archive "$ARCHIVE"
finish "$TARGET"
