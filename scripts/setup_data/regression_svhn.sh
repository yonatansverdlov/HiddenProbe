#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NAME="SVHN"
TARGET="$DATA_ROOT/regression/svhn"
ARCHIVE="$DOWNLOAD_DIR/svhn_cropped.tar.xz"
SPLIT="$TARGET/split.csv"

ARCHIVE_URL="https://storage.googleapis.com/gresearch/smallcnnzoo-dataset/svhn_cropped.tar.xz"
SPLIT_URL="https://raw.githubusercontent.com/AllanYangZhou/nfn/main/experiments/predict_gen_data_splits/svhn_split.csv"

log "============================================================"
log "SVHN regression dataset setup"
log "Target directory: $TARGET"
log "Archive path:     $ARCHIVE"
log "============================================================"

require_cmd tar

log "Ensuring target directory exists..."
mkdir -p "$TARGET"
log "Target directory ready."

log "Step 1: checking whether the dataset is already complete."
if dataset_files_ready \
    "$NAME" \
    "$TARGET" \
    "weights.npy" \
    "metrics.csv.gz" \
    "layout.csv" \
    "split.csv"; then
    log "Nothing else to do."
    exit 0
fi

log "Step 2: checking whether the model zoo is already extracted."
if [[ -s "$TARGET/weights.npy" \
   && -s "$TARGET/metrics.csv.gz" \
   && -s "$TARGET/layout.csv" ]]; then
    log "Extracted SVHN model-zoo files are present."

    if [[ ! -s "$SPLIT" ]]; then
        log "split.csv is missing."
        log "Only the split file needs to be downloaded."
        download_url "$SPLIT_URL" "$SPLIT"

        log "Re-checking complete dataset after split download..."
        if dataset_files_ready \
            "$NAME" \
            "$TARGET" \
            "weights.npy" \
            "metrics.csv.gz" \
            "layout.csv" \
            "split.csv"; then
            log "Nothing else to do."
            exit 0
        fi
        die "$NAME setup is still incomplete after downloading split.csv."
    fi
else
    log "Extracted model zoo is incomplete or missing."
fi

log "Step 3: checking for an existing downloaded archive."
if [[ -s "$ARCHIVE" ]]; then
    log "Existing archive found: $ARCHIVE"
    log "Skipping download and proceeding directly to extraction."
else
    log "No complete archive found."
    log "Downloading SVHN Small CNN Zoo..."
    download_url "$ARCHIVE_URL" "$ARCHIVE"
    log "SVHN archive download complete."
fi

log "Step 4: preparing a clean extraction workspace."
PARENT="$(dirname "$TARGET")"
TMP_PREFIX=".svhn_extract."
cleanup_stale_extract_dirs "$PARENT" "$TMP_PREFIX"

create_tmp() {
    log "Creating temporary extraction directory..."
    TMP="$(mktemp -d "$PARENT/${TMP_PREFIX}XXXXXX")"
    log "Temporary extraction directory: $TMP"
}

create_tmp
trap 'rm -rf "${TMP:-}"' EXIT

log "Step 5: Extracting SVHN Small CNN Zoo..."
log "Extraction may take some time."

if extract_tar_xz_with_progress "$ARCHIVE" "$TMP" "SVHN extraction"; then
    log "Extraction complete."
else
    log "Extraction failed."
    log "Assuming the existing archive is corrupted or incomplete."

    log "Removing failed extraction directory..."
    rm -rf "$TMP"
    log "Failed extraction directory removed."

    log "Deleting bad archive..."
    rm -f "$ARCHIVE" "${ARCHIVE}.part"
    log "Bad archive deleted."

    log "Downloading SVHN Small CNN Zoo again from scratch..."
    download_url "$ARCHIVE_URL" "$ARCHIVE"
    log "SVHN archive re-download complete."

    log "Preparing a fresh extraction directory..."
    create_tmp

    log "Retrying extraction..."
    if ! extract_tar_xz_with_progress "$ARCHIVE" "$TMP" "SVHN extraction retry"; then
        log "Second extraction attempt failed."
        log "Deleting the newly downloaded archive because it cannot be extracted."
        rm -f "$ARCHIVE" "${ARCHIVE}.part"
        die "SVHN extraction failed twice."
    fi

    log "Extraction complete on second attempt."
fi

log "Step 6: locating extracted dataset files."
WEIGHTS="$(find "$TMP" -type f -name weights.npy -print -quit)"
if [[ -z "$WEIGHTS" ]]; then
    die "Extraction completed, but weights.npy was not found."
fi
log "Found weights.npy at: $WEIGHTS"

SRC="$(dirname "$WEIGHTS")"
log "Detected extracted dataset directory: $SRC"

log "Checking extracted files before installation..."
require_file "$SRC/weights.npy"
require_file "$SRC/metrics.csv.gz"
require_file "$SRC/layout.csv"
log "All extracted model-zoo files passed sanity checks."

log "Step 7: installing extracted SVHN files into final target."
log "Removing any incomplete previous target files..."
rm -f "$TARGET/weights.npy" "$TARGET/metrics.csv.gz" "$TARGET/layout.csv"
log "Old incomplete target files removed."

log "Moving weights.npy..."
mv "$SRC/weights.npy" "$TARGET/weights.npy"
log "weights.npy installed."

log "Moving metrics.csv.gz..."
mv "$SRC/metrics.csv.gz" "$TARGET/metrics.csv.gz"
log "metrics.csv.gz installed."

log "Moving layout.csv..."
mv "$SRC/layout.csv" "$TARGET/layout.csv"
log "layout.csv installed."

log "Extracted SVHN model-zoo files installed successfully."

log "Step 8: checking split.csv."
if [[ -s "$SPLIT" ]]; then
    log "split.csv already exists."
else
    log "split.csv is missing."
    log "Downloading SVHN split.csv..."
    download_url "$SPLIT_URL" "$SPLIT"
    log "split.csv download complete."
fi

log "Step 9: running final dataset sanity check."
if ! dataset_files_ready \
    "$NAME" \
    "$TARGET" \
    "weights.npy" \
    "metrics.csv.gz" \
    "layout.csv" \
    "split.csv"; then
    die "SVHN setup finished, but one or more required files are missing."
fi
log "Final dataset sanity check passed."

log "Step 10: cleaning up downloaded archive."
cleanup_archive "$ARCHIVE"

finish "$TARGET"
