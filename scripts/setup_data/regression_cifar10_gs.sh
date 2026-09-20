#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NAME="CIFAR10-GS"
TARGET="$DATA_ROOT/regression/cifar10_gs"
ARCHIVE="$DOWNLOAD_DIR/cifar10.tar.xz"
ARCHIVE_URL="https://storage.googleapis.com/gresearch/smallcnnzoo-dataset/cifar10.tar.xz"
SPLIT_SRC="$SCRIPT_DIR/splits/gs_splits/nfn_cifar10_split.csv"     # shipped with the repository
SPLIT="$TARGET/nfn_cifar10_split.csv"

log "============================================================"
log "CIFAR10-GS regression dataset setup (Small CNN Zoo)"
log "Target directory: $TARGET"
log "Archive path:     $ARCHIVE"
log "============================================================"

require_cmd tar
mkdir -p "$TARGET"

install_split() {
    require_file "$SPLIT_SRC"
    log "Installing shipped split file: $SPLIT_SRC -> $SPLIT"
    cp -f "$SPLIT_SRC" "$SPLIT"
}

log "Step 1: checking whether the dataset is already complete."
if dataset_files_ready "$NAME" "$TARGET" "weights.npy" "metrics.csv.gz" "layout.csv" "nfn_cifar10_split.csv"; then
    log "Nothing else to do."
    exit 0
fi

log "Step 2: checking whether the model zoo is already extracted."
if [[ -s "$TARGET/weights.npy" && -s "$TARGET/metrics.csv.gz" && -s "$TARGET/layout.csv" ]]; then
    log "Extracted CIFAR10-GS model-zoo files are present."
    [[ -s "$SPLIT" ]] || install_split
    if dataset_files_ready "$NAME" "$TARGET" "weights.npy" "metrics.csv.gz" "layout.csv" "nfn_cifar10_split.csv"; then
        log "Nothing else to do."
        exit 0
    fi
    die "$NAME setup is still incomplete after installing the split file."
else
    log "Extracted model zoo is incomplete or missing."
fi

log "Step 3: checking for an existing downloaded archive."
if [[ -s "$ARCHIVE" ]]; then
    log "Existing archive found: $ARCHIVE"
else
    log "Downloading CIFAR10-GS Small CNN Zoo..."
    download_url "$ARCHIVE_URL" "$ARCHIVE"
fi

log "Step 4: preparing a clean extraction workspace."
PARENT="$(dirname "$TARGET")"
TMP_PREFIX=".cifar10_gs_extract."
cleanup_stale_extract_dirs "$PARENT" "$TMP_PREFIX"
create_tmp() {
    TMP="$(mktemp -d "$PARENT/${TMP_PREFIX}XXXXXX")"
    log "Temporary extraction directory: $TMP"
}
create_tmp
trap 'rm -rf "${TMP:-}"' EXIT

log "Step 5: extracting CIFAR10-GS Small CNN Zoo..."
if ! extract_tar_xz_with_progress "$ARCHIVE" "$TMP" "cifar10_gs extraction"; then
    log "Extraction failed; assuming the archive is corrupted or incomplete."
    rm -rf "$TMP"; rm -f "$ARCHIVE" "${ARCHIVE}.part"
    log "Downloading CIFAR10-GS Small CNN Zoo again from scratch..."
    download_url "$ARCHIVE_URL" "$ARCHIVE"
    create_tmp
    if ! extract_tar_xz_with_progress "$ARCHIVE" "$TMP" "cifar10_gs extraction retry"; then
        rm -f "$ARCHIVE" "${ARCHIVE}.part"
        die "CIFAR10-GS extraction failed twice."
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

log "Step 8: installing the split file."
[[ -s "$SPLIT" ]] || install_split

log "Step 9: running final dataset sanity check."
dataset_files_ready "$NAME" "$TARGET" "weights.npy" "metrics.csv.gz" "layout.csv" "nfn_cifar10_split.csv" || die "$NAME setup finished, but one or more required files are missing."

log "Step 10: cleaning up downloaded archive."
cleanup_archive "$ARCHIVE"

finish "$TARGET"
