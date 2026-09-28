#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NAME="CIFAR10-GS"
TARGET="$DATA_ROOT/regression/cifar10_gs"
ARCHIVE="$DOWNLOAD_DIR/cifar10.tar.xz"
SPLIT="$TARGET/split.csv"

ARCHIVE_URL="https://storage.googleapis.com/gresearch/smallcnnzoo-dataset/cifar10.tar.xz"
SPLIT_URL="https://raw.githubusercontent.com/AllanYangZhou/nfn/main/experiments/predict_gen_data_splits/cifar10_split.csv"

log "============================================================"
log "CIFAR10-GS regression dataset setup (Small CNN Zoo)"
log "Target directory: $TARGET"
log "Archive path:     $ARCHIVE"
log "Split:            official NFN split.csv"
log "============================================================"

require_cmd tar
mkdir -p "$TARGET"

install_split() {
    if [[ -s "$SPLIT" ]]; then
        log "Official NFN split already exists: $SPLIT"
        return
    fi
    log "Downloading official NFN CIFAR10 split -> $SPLIT"
    download_url "$SPLIT_URL" "$SPLIT"
}

log "Step 1: checking whether the dataset is already complete."
if dataset_files_ready "$NAME" "$TARGET" "weights.npy" "metrics.csv.gz" "layout.csv" "split.csv"; then
    log "Nothing else to do."
    exit 0
fi

log "Step 2: checking whether the model zoo is already extracted."
if [[ -s "$TARGET/weights.npy" && -s "$TARGET/metrics.csv.gz" && -s "$TARGET/layout.csv" ]]; then
    log "Extracted CIFAR10-GS model-zoo files are present."
    install_split
    dataset_files_ready "$NAME" "$TARGET" "weights.npy" "metrics.csv.gz" "layout.csv" "split.csv"         || die "$NAME setup is still incomplete after installing the official split."
    log "Nothing else to do."
    exit 0
fi

log "Step 3: checking for an existing downloaded archive."
if [[ -s "$ARCHIVE" ]]; then
    log "Existing archive found: $ARCHIVE"
else
    log "Downloading CIFAR10-GS Small CNN Zoo..."
    download_url "$ARCHIVE_URL" "$ARCHIVE"
fi

log "Step 4: extracting CIFAR10-GS Small CNN Zoo directly into final target."
rm -rf "$TARGET"
mkdir -p "$TARGET"

if ! extract_tar_xz_with_progress "$ARCHIVE" "$TARGET" "cifar10_gs extraction"; then
    log "Extraction failed; assuming the archive is corrupted or incomplete."
    rm -rf "$TARGET"
    rm -f "$ARCHIVE" "${ARCHIVE}.part"
    log "Downloading CIFAR10-GS Small CNN Zoo again from scratch..."
    download_url "$ARCHIVE_URL" "$ARCHIVE"

    mkdir -p "$TARGET"
    if ! extract_tar_xz_with_progress "$ARCHIVE" "$TARGET" "cifar10_gs extraction retry"; then
        rm -rf "$TARGET"
        rm -f "$ARCHIVE" "${ARCHIVE}.part"
        die "CIFAR10-GS extraction failed twice."
    fi
fi
log "Extraction complete."

log "Step 6: locating extracted dataset files."
WEIGHTS="$(find "$TARGET" -type f -name weights.npy -print -quit)"
[[ -n "$WEIGHTS" ]] || die "Extraction completed, but weights.npy was not found."
SRC="$(dirname "$WEIGHTS")"
require_file "$SRC/weights.npy"
require_file "$SRC/metrics.csv.gz"
require_file "$SRC/layout.csv"

log "Step 7: normalizing extracted files in $TARGET."
if [[ "$SRC" != "$TARGET" ]]; then
    mv "$SRC/weights.npy" "$TARGET/weights.npy"
    mv "$SRC/metrics.csv.gz" "$TARGET/metrics.csv.gz"
    mv "$SRC/layout.csv" "$TARGET/layout.csv"
fi

log "Step 8: installing the official NFN split."
install_split

log "Step 9: running final dataset sanity check."
dataset_files_ready "$NAME" "$TARGET" "weights.npy" "metrics.csv.gz" "layout.csv" "split.csv"     || die "$NAME setup finished, but one or more required files are missing."

log "Step 10: cleaning up downloaded archive."
cleanup_archive "$ARCHIVE"

finish "$TARGET"
