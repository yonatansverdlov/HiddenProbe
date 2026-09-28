#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NAME="AGNews-Transformers"
TARGET="$DATA_ROOT/ag_news_transformer"
ARCHIVE="$DOWNLOAD_DIR/AG-News-Transformers.zip"
URL="https://huggingface.co/datasets/anonymized-acamedia/Small-Transformer-Zoo/resolve/main/AG-News-Transformers.zip"

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

log "Extracting $NAME zoo directly into final target."
rm -rf "$TARGET"
mkdir -p "$TARGET"
if ! extract_zip_with_progress "$ARCHIVE" "$TARGET" "$NAME extraction"; then
    rm -rf "$TARGET"
    rm -f "$ARCHIVE" "$ARCHIVE.part"
    die "$NAME extraction failed."
fi

if ! find "$TARGET" -type f -name '*_75_*.pt' -print -quit 2>/dev/null | grep -q .; then
    die "$NAME installed, but no epoch-75 checkpoint was found."
fi

cleanup_archive "$ARCHIVE"
finish "$TARGET"
