#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NAME="CIFAR100 INR"
TARGET="$DATA_ROOT/classification/cifar100_inr"
STAGING="$DOWNLOAD_DIR/cifar100_inr_dng"
SPLIT="$TARGET/dng_cifar100_split_noaug.json"

# Official pretrained CIFAR-100 SIREN zoo linked by:
# Wu et al., "Dynamic Neural Graph Encoding of Inference Processes in Deep Weight Space"
# arXiv:2607.02166 / TMLR 2026.
GDRIVE_FOLDER_ID="1TwUZmcE2XrGQXCPhGAIa_sXCd5kX8OIA"
GDRIVE_FOLDER_URL="https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}"

log "============================================================"
log "CIFAR100 INR classification dataset setup (DNG/TMLR 2026)"
log "Target directory: $TARGET"
log "Source: $GDRIVE_FOLDER_URL"
log "============================================================"

find_zoo_root() {
    local base="$1"
    local named
    named="$(find "$base" -type d -name 'siren_cifar100_wts' -print -quit 2>/dev/null || true)"
    if [[ -n "$named" ]]; then
        echo "$named"
        return 0
    fi

    local sample parent
    sample="$(find "$base" -type d -name 'randinit_smaller_0s' -print -quit 2>/dev/null || true)"
    if [[ -n "$sample" ]]; then
        parent="$(dirname "$sample")"
        echo "$parent"
        return 0
    fi
    return 1
}

build_split() {
    if [[ -s "$SPLIT" ]]; then
        log "CIFAR100 non-aug split already exists: $SPLIT"
        return
    fi

    require_cmd python
    log "Building DNG CIFAR100 non-aug 45k/5k/10k split..."
    python - "$TARGET" "$SPLIT" <<'PY'
import glob
import json
import os
import re
import sys

root, out_path = sys.argv[1:]
idx_re = re.compile(r"net(\d+)\.(?:pth|pt)$")
label_re = re.compile(r"_(\d+)s$")

by_idx = {}
for d in glob.glob(os.path.join(root, "randinit_smaller_*")):
    if not os.path.isdir(d):
        continue
    dirname = os.path.basename(d)
    # Use the original/non-augmented zoo only.
    if "aug" in dirname.lower():
        continue
    lm = label_re.search(dirname)
    if lm is None:
        continue
    label = int(lm.group(1))
    for p in glob.glob(os.path.join(d, "net*.*")):
        m = idx_re.search(os.path.basename(p))
        if m is None:
            continue
        idx = int(m.group(1))
        rel = os.path.relpath(p, root)
        if idx in by_idx:
            raise RuntimeError(f"Duplicate non-aug INR index {idx}: {by_idx[idx][0]} and {rel}")
        by_idx[idx] = (rel, label)

expected = 60000
missing = [i for i in range(expected) if i not in by_idx]
if missing:
    raise RuntimeError(
        f"Expected CIFAR100 INR indices 0..59999; found {len(by_idx)} unique models. "
        f"First missing indices: {missing[:20]}"
    )

splits = {s: {"path": [], "label": []} for s in ("train", "val", "test")}
for idx in range(expected):
    split = "train" if idx < 45000 else ("val" if idx < 50000 else "test")
    rel, label = by_idx[idx]
    splits[split]["path"].append(rel)
    splits[split]["label"].append(label)

os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(splits, f)

print(f"[cifar100-split] wrote {out_path}")
for split in ("train", "val", "test"):
    labels = splits[split]["label"]
    print(f"  {split}: {len(labels)} models; classes={len(set(labels))}")
PY
}

mkdir -p "$TARGET"

if find "$TARGET" -type f \( -name 'net*.pth' -o -name 'net*.pt' \) -print -quit 2>/dev/null | grep -q .; then
    log "$NAME weights already exist at $TARGET; skipping download."
    build_split
    require_file "$SPLIT"
    finish "$TARGET"
    exit 0
fi

if command -v gdown >/dev/null 2>&1; then
    GDOWN=(gdown)
elif python -c 'import gdown' >/dev/null 2>&1; then
    GDOWN=(python -m gdown)
else
    die "CIFAR100 source is a Google Drive folder. Install gdown first: python -m pip install gdown"
fi

rm -rf "$STAGING"
mkdir -p "$STAGING"
log "Downloading the official CIFAR100 SIREN folder from the paper's Google Drive..."
"${GDOWN[@]}" --folder "$GDRIVE_FOLDER_URL" -O "$STAGING"

SRC="$(find_zoo_root "$STAGING" || true)"
if [[ -z "$SRC" ]]; then
    # The Drive folder may contain an archive rather than the unpacked zoo.
    EXTRACTED="$STAGING/_extracted"
    mkdir -p "$EXTRACTED"
    while IFS= read -r -d '' archive; do
        case "$archive" in
            *.zip)
                require_cmd unzip
                unzip -q "$archive" -d "$EXTRACTED"
                ;;
            *.tar|*.tar.gz|*.tgz|*.tar.xz)
                require_cmd tar
                tar -xf "$archive" -C "$EXTRACTED"
                ;;
        esac
    done < <(find "$STAGING" -type f \( -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.tar.xz' \) -print0)
    SRC="$(find_zoo_root "$EXTRACTED" || true)"
fi

[[ -n "$SRC" ]] || die "Downloaded the DNG CIFAR100 folder but could not locate siren_cifar100_wts/randinit_smaller_*."

log "Installing CIFAR100 SIREN weights from: $SRC"
rm -rf "$TARGET"
mkdir -p "$TARGET"
cp -a "$SRC"/. "$TARGET"/

find "$TARGET" -type f \( -name 'net*.pth' -o -name 'net*.pt' \) -print -quit 2>/dev/null | grep -q . \
    || die "CIFAR100 INR install completed, but no net*.pth/net*.pt checkpoints were found."

build_split
require_file "$SPLIT"

if [[ "${KEEP_ARCHIVES:-0}" != "1" ]]; then
    rm -rf "$STAGING"
fi

finish "$TARGET"
