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

EXPECTED_PTH=70000

valid_mnist_tree() {
    local root="$1"
    local mnist_root="$root/mnist-inrs"

    [[ -d "$mnist_root" ]] || return 1

    # MNIST-only guard: reject archives/installations containing other datasets.
    if find "$mnist_root" -print 2>/dev/null | grep -Eqi '/(cifar|fmnist|fashion)[^/]*'; then
        return 1
    fi

    local n_pth
    n_pth="$(find "$mnist_root" -type f -name '*.pth' 2>/dev/null | wc -l | tr -d ' ')"
    [[ "$n_pth" -eq "$EXPECTED_PTH" ]]
}

log "============================================================"
log "MNIST INR classification dataset setup"
log "Target: $TARGET"
log "============================================================"

require_cmd unzip
mkdir -p "$TARGET"

log "Checking whether MNIST INR is already complete."
if [[ -s "$SPLIT" ]] && valid_mnist_tree "$DATASET_DIR"; then
    log "$NAME is already installed; nothing to do."
    exit 0
fi

if [[ -d "$DATASET_DIR" ]]; then
    log "Existing MNIST INR directory is incomplete or contains non-MNIST data; replacing it."
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

log "Extracting MNIST INR archive."
if ! extract_zip_with_progress "$ARCHIVE" "$TMP" "MNIST INR extraction"; then
    log "Extraction failed; treating the archive as corrupt/incomplete."
    rm -rf "$TMP"
    rm -f "$ARCHIVE" "${ARCHIVE}.part"

    log "Re-downloading archive from scratch."
    download_url "$ARCHIVE_URL" "$ARCHIVE"

    create_tmp
    if ! extract_zip_with_progress "$ARCHIVE" "$TMP" "MNIST INR extraction"; then
        rm -f "$ARCHIVE" "${ARCHIVE}.part"
        die "MNIST INR extraction failed twice."
    fi
fi

MNIST_ROOT="$(find "$TMP" -type d -name 'mnist-inrs' -print -quit)"
[[ -n "$MNIST_ROOT" ]] || die "Archive does not contain the expected mnist-inrs directory."

if find "$MNIST_ROOT" -print | grep -Eqi '/(cifar|fmnist|fashion)[^/]*'; then
    die "Downloaded archive contains non-MNIST paths; refusing to install it."
fi

PTH_COUNT="$(find "$MNIST_ROOT" -type f -name '*.pth' | wc -l | tr -d ' ')"
[[ "$PTH_COUNT" -eq "$EXPECTED_PTH" ]] \
    || die "Expected $EXPECTED_PTH MNIST INR checkpoints, found $PTH_COUNT."

log "Validated MNIST-only archive: $PTH_COUNT checkpoints."
log "Installing only mnist-inrs into canonical data directory."
rm -rf "$DATASET_DIR"
mkdir -p "$DATASET_DIR"
cp -a "$MNIST_ROOT" "$DATASET_DIR/"

valid_mnist_tree "$DATASET_DIR" \
    || die "Installed MNIST INR tree failed the final MNIST-only sanity check."

if [[ -s "$SPLIT" ]]; then
    log "Split JSON already exists."
else
    log "Downloading MNIST split JSON."
    download_url "$SPLIT_URL" "$SPLIT"
fi

require_file "$SPLIT"
valid_mnist_tree "$DATASET_DIR" \
    || die "Final sanity check failed: MNIST-only INR tree is invalid."

cleanup_archive "$ARCHIVE"
finish "$TARGET"
