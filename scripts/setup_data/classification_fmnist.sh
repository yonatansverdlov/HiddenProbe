#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NAME="Fashion-MNIST INR"
TARGET="$DATA_ROOT/classification/fmnist_inr"
DATASET_DIR="$TARGET/dataset"
ARCHIVE="$DOWNLOAD_DIR/fmnist_inrs.zip"
SPLIT="$TARGET/fmnist_splits.json"

ARCHIVE_URL="https://www.dropbox.com/sh/56pakaxe58z29mq/AAAssoHq719OmSHSKKTiKKHGa/fmnist_inrs.zip?dl=1"
SPLIT_URL="https://raw.githubusercontent.com/jonkahana/ProbeGen/main/experiments/inr_classification/dataset/fmnist_splits.json"


extract_fmnist_checkpoints_with_progress() {
    local archive="$1"
    local dest="$2"
    local train_pattern='fmnist_inrs/train/model_*.pth'
    local test_pattern='fmnist_inrs/test/model_*.pth'

    local total
    total="$(unzip -Z1 "$archive" "$train_pattern" "$test_pattern" 2>/dev/null | wc -l | tr -d '[:space:]')"
    [[ -n "$total" && "$total" -gt 0 ]] || return 1
    log "Fashion-MNIST INR checkpoints: $total files"

    local awk_program='
        BEGIN { width=40; count=0; last=-1 }
        /(^|[[:space:]])(inflating:|extracting:)/ {
            count++
            pct=int((count*100)/total)
            if (pct>100) pct=100
            if (pct != last) {
                filled=int(width*pct/100)
                bar=""
                for (i=0; i<filled; i++) bar=bar "#"
                for (i=filled; i<width; i++) bar=bar "-"
                printf "\r[setup] Fashion-MNIST INR extraction: [%s] %3d%% (%d/%d)", bar, pct, count, total > "/dev/stderr"
                fflush("/dev/stderr")
                last=pct
            }
        }
        END { if (count > 0) printf "\n" > "/dev/stderr" }
    '

    unzip -o "$archive" "$train_pattern" "$test_pattern" -d "$dest" 2>&1 \
        | awk -v total="$total" "$awk_program"
}

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

log "Extracting Fashion-MNIST INR checkpoints directly into final data directory."
rm -rf "$DATASET_DIR"
mkdir -p "$DATASET_DIR"

if ! extract_fmnist_checkpoints_with_progress "$ARCHIVE" "$DATASET_DIR"; then
    log "Extraction failed; treating the archive as corrupt/incomplete."
    rm -rf "$DATASET_DIR"
    rm -f "$ARCHIVE" "${ARCHIVE}.part"

    log "Re-downloading archive from scratch."
    download_url "$ARCHIVE_URL" "$ARCHIVE"

    mkdir -p "$DATASET_DIR"
    if ! extract_fmnist_checkpoints_with_progress "$ARCHIVE" "$DATASET_DIR"; then
        rm -rf "$DATASET_DIR"
        rm -f "$ARCHIVE" "${ARCHIVE}.part"
        die "Fashion-MNIST INR extraction failed twice."
    fi
fi

find "$DATASET_DIR" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q . \
    || die "No Fashion-MNIST INR checkpoints found after extraction."

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
