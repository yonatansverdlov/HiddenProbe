#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NAME="MNIST-Transformers"
TARGET="$DATA_ROOT/mnist_transformer"
ARCHIVE="$DOWNLOAD_DIR/MNIST-Transformers.zip"
URL="https://huggingface.co/datasets/anonymized-acamedia/Small-Transformer-Zoo/resolve/main/MNIST-Transformers.zip"

log "============================================================"
log "$NAME setup"
log "Target directory: $TARGET"
log "Archive path:     $ARCHIVE"
log "============================================================"

if [[ -d "$TARGET" ]] && find "$TARGET" -type f -name '*_75_*.pt' -print -quit 2>/dev/null | grep -q .; then
    log "$NAME zoo already exists at: $TARGET"
    finish "$TARGET"
    exit 0
fi

require_cmd unzip
mkdir -p "$TARGET"

if [[ ! -s "$ARCHIVE" ]]; then
    log "Downloading $NAME zoo..."
    download_url "$URL" "$ARCHIVE"
else
    log "Using existing archive: $ARCHIVE"
fi

PARENT="$(dirname "$TARGET")"
TMP_PREFIX=".mnist_transformer_extract."
cleanup_stale_extract_dirs "$PARENT" "$TMP_PREFIX"
TMP="$(mktemp -d "$PARENT/${TMP_PREFIX}XXXXXX")"
trap 'rm -rf "${TMP:-}"' EXIT

log "Extracting $NAME zoo..."
if ! extract_zip_with_progress "$ARCHIVE" "$TMP" "$NAME extraction"; then
    rm -rf "$TMP"
    rm -f "$ARCHIVE" "$ARCHIVE.part"
    die "$NAME extraction failed."
fi

rm -rf "$TARGET"
mkdir -p "$TARGET"
cp -a "$TMP"/. "$TARGET"/

if ! find "$TARGET" -type f -name '*_75_*.pt' -print -quit 2>/dev/null | grep -q .; then
    die "$NAME installed, but no epoch-75 checkpoint was found."
fi

cleanup_archive "$ARCHIVE"
finish "$TARGET"
