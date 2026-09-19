#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NAME="MNIST-Transformers"
DATASET="mnist"                                   # --dataset value for `main.py transformer`
ARCHIVE_NAME="MNIST-Transformers.zip"             # file in the HuggingFace dataset repo
TARGET="$DATA_ROOT/mnist_transformer"             # wrapper dir the transformer loader looks under (recursively) for <run>_75_<acc>.pt
SPLIT_SEED="${SPLIT_SEED:-0}"

log "============================================================"
log "$NAME zoo setup (Small Transformer Zoo, HuggingFace: $HF_REPO)"
log "Data root:        $DATA_ROOT"
log "Target directory: $TARGET"
log "Archive:          $DOWNLOAD_DIR/$ARCHIVE_NAME"
log "============================================================"

log "Step 1: checkpoints."
if ! zoo_ready "$NAME" "$TARGET"; then
    install_zoo "$NAME" "$ARCHIVE_NAME" "$TARGET"
    zoo_ready "$NAME" "$TARGET" || die "$NAME installed, but no epoch-75 checkpoint was found under $TARGET."
    cleanup_archive "$DOWNLOAD_DIR/$ARCHIVE_NAME"
fi

log "Step 2: consolidated per-split cache (epoch-75 checkpoints, seeded run-disjoint train/val/test split)."
if ! cache_ready "$DATASET" "$SPLIT_SEED"; then
    require_cmd "$PYTHON"
    log "Building the cache once (loads every checkpoint; this takes a while)..."
    (cd "$REPO_ROOT" && "$PYTHON" main.py transformer cache --dataset "$DATASET" --seed "$SPLIT_SEED" --data_root "$DATA_ROOT")
    cache_ready "$DATASET" "$SPLIT_SEED" || die "$NAME cache build finished, but the cache files are missing."
fi

finish "$TARGET"
