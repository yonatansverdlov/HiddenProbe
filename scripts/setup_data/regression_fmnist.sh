#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NAME="Fashion-MNIST"
TARGET="$DATA_ROOT/regression/fmnist"
ARCHIVE="$DOWNLOAD_DIR/fashion_mnist.tar.xz"
ARCHIVE_URL="https://storage.googleapis.com/gresearch/smallcnnzoo-dataset/fashion_mnist.tar.xz"

log "============================================================"
log "Fashion-MNIST regression dataset setup (Small CNN Zoo)"
log "Target directory: $TARGET"
log "Archive path:     $ARCHIVE"
log "============================================================"

require_cmd tar
mkdir -p "$TARGET"

log "Step 1: checking whether the dataset is already complete."
if dataset_files_ready "$NAME" "$TARGET" "weights.npy" "metrics.csv.gz" "layout.csv"; then
    log "Nothing else to do."
    exit 0
fi

log "Step 2: checking whether the model zoo is already extracted."
if [[ -s "$TARGET/weights.npy" && -s "$TARGET/metrics.csv.gz" && -s "$TARGET/layout.csv" ]]; then
    log "Extracted Fashion-MNIST model-zoo files are present."
    die "$NAME files are present but the readiness check failed; inspect $TARGET."
else
    log "Extracted model zoo is incomplete or missing."
fi

log "Step 3: checking for an existing downloaded archive."
if [[ -s "$ARCHIVE" ]]; then
    log "Existing archive found: $ARCHIVE"
else
    log "Downloading Fashion-MNIST Small CNN Zoo..."
    download_url "$ARCHIVE_URL" "$ARCHIVE"
fi

log "Step 4: preparing a clean extraction workspace."
PARENT="$(dirname "$TARGET")"
TMP_PREFIX=".fmnist_extract."
cleanup_stale_extract_dirs "$PARENT" "$TMP_PREFIX"
create_tmp() {
    TMP="$(mktemp -d "$PARENT/${TMP_PREFIX}XXXXXX")"
    log "Temporary extraction directory: $TMP"
}
create_tmp
trap 'rm -rf "${TMP:-}"' EXIT

log "Step 5: extracting Fashion-MNIST Small CNN Zoo..."
if ! extract_tar_xz_with_progress "$ARCHIVE" "$TMP" "fmnist extraction"; then
    log "Extraction failed; assuming the archive is corrupted or incomplete."
    rm -rf "$TMP"; rm -f "$ARCHIVE" "${ARCHIVE}.part"
    log "Downloading Fashion-MNIST Small CNN Zoo again from scratch..."
    download_url "$ARCHIVE_URL" "$ARCHIVE"
    create_tmp
    if ! extract_tar_xz_with_progress "$ARCHIVE" "$TMP" "fmnist extraction retry"; then
        rm -f "$ARCHIVE" "${ARCHIVE}.part"
        die "Fashion-MNIST extraction failed twice."
    fi
fi
log "Extraction complete."

log "Step 6: locating extracted dataset files."
WEIGHTS="$(find "$TMP" -type f -name weights.npy -print -quit)"
[[ -n "$WEIGHTS" ]] || die "Extraction completed, but weights.npy was not found."
SRC="$(dirname "$WEIGHTS")"
require_file "$SRC/weights.npy"; require_file "$SRC/metrics.csv.gz"; require_file "$SRC/layout.csv"

log "Step 7: installing extracted files into $TARGET."
rm -f "$TARGET/weights.npy" "$TARGET/metrics.csv.gz" "$TARGET/layout.csv"
mv "$SRC/weights.npy" "$TARGET/weights.npy"
mv "$SRC/metrics.csv.gz" "$TARGET/metrics.csv.gz"
mv "$SRC/layout.csv" "$TARGET/layout.csv"

log "Step 8: no split file is shipped for this zoo; the loader generates the seed-0 split deterministically on first use."

log "Step 9: running final dataset sanity check."
dataset_files_ready "$NAME" "$TARGET" "weights.npy" "metrics.csv.gz" "layout.csv" || die "$NAME setup finished, but one or more required files are missing."

log "Step 10: cleaning up downloaded archive."
cleanup_archive "$ARCHIVE"

finish "$TARGET"
