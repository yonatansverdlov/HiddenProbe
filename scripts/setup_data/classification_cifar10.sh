#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NAME="CIFAR10 INR"
TARGET="$DATA_ROOT/classification/cifar10_inr"
ARCHIVE="$DOWNLOAD_DIR/siren_cifar_wts.tar"

# NFN CIFAR-10 SIREN zoo (arXiv:2305.13546), the `siren_cifar_wts` release on Google Drive.
GDRIVE_FILE_ID="14RUV3eN6-lSOr9XuwyKFQFVcqKl0L2bw"

# Split files (INRDataset format) built from the extracted zoo, next to the checkpoints:
#   nfn_cifar_split.json        augmented   (train = base + 20 random-init copies per image)
#   nfn_cifar_split_noaug.json  non-augmented (one SIREN per image everywhere)
SPLIT_AUG="$TARGET/nfn_cifar_split.json"
SPLIT_NOAUG="$TARGET/nfn_cifar_split_noaug.json"

log "============================================================"
log "CIFAR10 INR classification dataset setup"
log "Target directory: $TARGET"
log "Archive path:     $ARCHIVE"
log "============================================================"

require_cmd tar

build_splits() {
    require_cmd python

    if [[ -s "$SPLIT_AUG" && -s "$SPLIT_NOAUG" ]]; then
        log "Both NFN split files already exist."
        return
    fi

    log "Building augmented and non-augmented NFN split files from the extracted SIREN zoo..."
    python - "$TARGET" "$SPLIT_AUG" "$SPLIT_NOAUG" <<'PY'
import glob
import json
import os
import re
import sys
from collections import defaultdict

siren_dir, aug_out, noaug_out = sys.argv[1:]
prefix = "randinit_smaller"
val_point = 45000
test_point = 50000

idx_re = re.compile(r"net(\d+)\.pth$")
lbl_re = re.compile(r"_(\d+)s")

files = glob.glob(os.path.join(siren_dir, f"{prefix}_*", "net*.pth"))
print(f"[nfn-split] globbed {len(files)} .pth files")

all_paths = defaultdict(list)
base_paths = defaultdict(list)
labels = {}

for path in files:
    dirname = os.path.basename(os.path.dirname(path))
    idx_match = idx_re.search(os.path.basename(path))
    label_match = lbl_re.search(dirname)
    if not (idx_match and label_match):
        continue

    idx = int(idx_match.group(1))
    label = int(label_match.group(1))
    rel = os.path.relpath(path, siren_dir)

    all_paths[idx].append(rel)
    labels[idx] = label
    if "aug" not in dirname:
        base_paths[idx].append(rel)

def split_for(idx):
    if idx < val_point:
        return "train"
    if idx < test_point:
        return "val"
    return "test"

def write_split(path_map, out_path, augmented):
    out = {s: {"path": [], "label": []} for s in ("train", "val", "test")}

    for idx in sorted(path_map):
        split = split_for(idx)
        copies = sorted(path_map[idx])
        chosen = copies if (augmented and split == "train") else copies[:1]
        for rel in chosen:
            out[split]["path"].append(rel)
            out[split]["label"].append(labels[idx])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)

    print(f"[nfn-split] wrote {out_path}")
    for split in ("train", "val", "test"):
        print(f"  {split}: {len(out[split]['label'])}")

if not os.path.isfile(aug_out) or os.path.getsize(aug_out) == 0:
    write_split(all_paths, aug_out, augmented=True)

if not os.path.isfile(noaug_out) or os.path.getsize(noaug_out) == 0:
    write_split(base_paths, noaug_out, augmented=False)
PY
}

log "Ensuring target directory exists..."
mkdir -p "$TARGET"
log "Target directory ready."

log "Step 1: checking whether the dataset is already complete."
if find "$TARGET" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q .; then
    log "Found CIFAR10 INR checkpoint files."
    log "$NAME dataset already exists at: $TARGET"
    log "Skipping download and extraction."
    build_splits
    log "Nothing else to do."
    exit 0
fi
log "$NAME dataset is incomplete or missing."

log "Step 2: checking for an existing downloaded archive."
if [[ -s "$ARCHIVE" ]]; then
    log "Existing archive found: $ARCHIVE"
    log "Skipping download and proceeding directly to extraction."
else
    log "No complete archive found."
    log "Downloading CIFAR10 SIREN weights from Google Drive..."
    download_gdrive "$GDRIVE_FILE_ID" "$ARCHIVE"
    log "CIFAR10 INR archive download complete."
fi

log "Step 3: extracting CIFAR10 SIREN weights directly into final target."
rm -rf "$TARGET"
mkdir -p "$TARGET"
log "Extraction may take some time."

if ! tar -xf "$ARCHIVE" -C "$TARGET"; then
    log "Extraction failed; assuming the archive is corrupted or incomplete."
    rm -rf "$TARGET"
    rm -f "$ARCHIVE" "${ARCHIVE}.part"

    log "Downloading CIFAR10 SIREN weights again from scratch..."
    download_gdrive "$GDRIVE_FILE_ID" "$ARCHIVE"

    mkdir -p "$TARGET"
    if ! tar -xf "$ARCHIVE" -C "$TARGET"; then
        rm -rf "$TARGET"
        rm -f "$ARCHIVE" "${ARCHIVE}.part"
        die "CIFAR10 INR extraction failed twice."
    fi
fi

# The archive may wrap the zoo in a siren_cifar_wts/ directory. Flatten it
# in-place so the canonical target contains randinit_smaller_* directly.
SRC="$TARGET/siren_cifar_wts"
if [[ -d "$SRC" ]]; then
    shopt -s dotglob nullglob
    items=("$SRC"/*)
    if (( ${#items[@]} > 0 )); then
        mv -- "${items[@]}" "$TARGET"/
    fi
    shopt -u dotglob nullglob
    rmdir "$SRC" 2>/dev/null || true
fi

find "$TARGET" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q . \
    || die "Extraction completed, but no .pth checkpoints were found."

log "Step 7: running final dataset sanity check."
find "$TARGET" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q . \
    || die "Final sanity check failed: no CIFAR10 INR checkpoints found."
log "Final dataset sanity check passed."

log "Step 8: building the augmented and non-augmented split files."
build_splits
require_file "$SPLIT_AUG"
require_file "$SPLIT_NOAUG"

log "Step 9: cleaning up downloaded archive."
cleanup_archive "$ARCHIVE"

finish "$TARGET"
