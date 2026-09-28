#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NAME="Fashion-MNIST"
TARGET="$DATA_ROOT/regression/fmnist"
ARCHIVE="$DOWNLOAD_DIR/fashion_mnist.tar.xz"
SPLIT="$TARGET/split.csv"

ARCHIVE_URL="https://storage.googleapis.com/gresearch/smallcnnzoo-dataset/fashion_mnist.tar.xz"
SPLIT_URL="https://raw.githubusercontent.com/AllanYangZhou/nfn/main/experiments/predict_gen_data_splits/fashion_mnist_split.csv"

log "============================================================"
log "Fashion-MNIST regression dataset setup (Small CNN Zoo)"
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
    log "Downloading official NFN Fashion-MNIST split -> $SPLIT"
    download_url "$SPLIT_URL" "$SPLIT"
}

log "Step 1: checking whether the dataset is already complete."
if dataset_files_ready "$NAME" "$TARGET" "weights.npy" "metrics.csv.gz" "layout.csv" "split.csv"; then
    log "Nothing else to do."
    exit 0
fi

log "Step 2: checking whether the model zoo is already extracted."
if [[ -s "$TARGET/weights.npy" && -s "$TARGET/metrics.csv.gz" && -s "$TARGET/layout.csv" ]]; then
    log "Extracted Fashion-MNIST model-zoo files are present."
    install_split
    dataset_files_ready "$NAME" "$TARGET" "weights.npy" "metrics.csv.gz" "layout.csv" "split.csv"         || die "$NAME setup is still incomplete after installing the official split."
    log "Nothing else to do."
    exit 0
fi

log "Step 3: checking for an existing downloaded archive."
if [[ -s "$ARCHIVE" ]]; then
    log "Existing archive found: $ARCHIVE"
else
    log "Downloading Fashion-MNIST Small CNN Zoo..."
    download_url "$ARCHIVE_URL" "$ARCHIVE"
fi

log "Step 4: extracting Fashion-MNIST Small CNN Zoo directly into final target."
rm -rf "$TARGET"
mkdir -p "$TARGET"

if ! extract_tar_xz_with_progress "$ARCHIVE" "$TARGET" "fmnist extraction"; then
    log "Extraction failed; assuming the archive is corrupted or incomplete."
    rm -rf "$TARGET"
    rm -f "$ARCHIVE" "${ARCHIVE}.part"
    log "Downloading Fashion-MNIST Small CNN Zoo again from scratch..."
    download_url "$ARCHIVE_URL" "$ARCHIVE"

    mkdir -p "$TARGET"
    if ! extract_tar_xz_with_progress "$ARCHIVE" "$TARGET" "fmnist extraction retry"; then
        rm -rf "$TARGET"
        rm -f "$ARCHIVE" "${ARCHIVE}.part"
        die "Fashion-MNIST extraction failed twice."
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
