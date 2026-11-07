#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NAME="CIFAR10 INR"
TARGET="$DATA_ROOT/classification/cifar10_inr"
ARCHIVE="$DOWNLOAD_DIR/siren_cifar_wts.tar"

# NFN CIFAR-10 SIREN zoo (arXiv:2305.13546), the `siren_cifar_wts` release on Google Drive.
GDRIVE_FILE_ID="14RUV3eN6-lSOr9XuwyKFQFVcqKl0L2bw"

# This setup is intentionally for the non-augmented CIFAR-10 INR task only.
# The upstream Google Drive link contains a single complete TAR (aug + base):
# we must download that TAR, but extract only the original SIREN checkpoints.
SPLIT_NOAUG="$TARGET/nfn_cifar_split_noaug.json"

log "============================================================"
log "CIFAR10 INR classification dataset setup"
log "Target directory: $TARGET"
log "Archive path:     $ARCHIVE"
log "============================================================"

require_cmd tar

# Select the one original INR per CIFAR-10 image. Archive members may be
# prefixed by "siren_cifar_wts/" or appear directly under randinit_smaller_*.
# We deliberately use the exact same "aug" naming convention as the split
# generator, to avoid keeping extra random-initialization augmentations.
extract_noaug_only() {
    local archive="$1"
    local dest="$2"
    local selected
    selected="$(mktemp)"
    if ! tar -tf "$archive" | python -c '
import pathlib
import re
import sys
count = 0
for line in sys.stdin:
    name = line.rstrip("\n")
    path = pathlib.PurePosixPath(name)
    if ".." in path.parts or path.is_absolute():
        continue
    dirname = path.parent.name
    if (path.suffix == ".pth"
            and re.fullmatch(r"net\d+\.pth", path.name)
            and dirname.startswith("randinit_smaller_")
            and "aug" not in dirname):
        print(name)
        count += 1
if count == 0:
    raise SystemExit("No original CIFAR-10 INR .pth files found in TAR")
' > "$selected"; then
        rm -f "$selected"
        return 1
    fi
    local count
    count="$(wc -l < "$selected")"
    log "Extracting $count original .pth checkpoints only (no augmentation)."
    # GNU tar -v prints one extracted filename per line to stdout.
    # Consume it as a progress counter instead of flooding the terminal.
    local status=0
    tar -xvf "$archive" -C "$dest" -T "$selected" | awk -v total="$count" '
        BEGIN {
            width = 40
            done = 0
            last_pct = -1
            printf "[setup] CIFAR-10 INR extraction: 0%% (0/%d)\n", total > "/dev/stderr"
        }
        {
            done++
            pct = int(100 * done / total)
            if (pct > 100) pct = 100
            if (pct != last_pct) {
                filled = int(width * pct / 100)
                bar = ""
                for (i = 0; i < filled; i++) bar = bar "#"
                for (i = filled; i < width; i++) bar = bar "-"
                printf "\r[setup] CIFAR-10 INR extraction: [%s] %3d%% (%d/%d)", bar, pct, done, total > "/dev/stderr"
                fflush("/dev/stderr")
                last_pct = pct
            }
        }
        END {
            printf "\n" > "/dev/stderr"
            fflush("/dev/stderr")
        }
    ' || status=$?
    rm -f "$selected"
    return "$status"
}

# The official no-augmentation data protocol uses 45k/5k/10k networks.
# A partial prior extraction must NOT be mistaken for a complete dataset.
have_original_checkpoints() {
    python - "$TARGET" <<'PY'
import glob
import os
import sys
root = sys.argv[1]
files = glob.glob(os.path.join(root, "randinit_smaller_*", "net*.pth"))
base_count = sum("aug" not in os.path.basename(os.path.dirname(p)) for p in files)
print(f"[setup] Original CIFAR-10 INR checkpoints present: {base_count}/60000")
sys.exit(0 if base_count == 60000 else 1)
PY
}

build_splits() {
    require_cmd python
    if [[ -s "$SPLIT_NOAUG" ]] && python - "$SPLIT_NOAUG" <<'PY'
import json
import sys
try:
    with open(sys.argv[1]) as handle:
        d = json.load(handle)
    assert [len(d[s]["path"]) for s in ("train", "val", "test")] == [45000, 5000, 10000]
except (OSError, ValueError, KeyError, AssertionError):
    raise SystemExit(1)
PY
    then
        log "Valid non-augmented split already exists."
        return
    fi

    log "Building ONLY the non-augmented CIFAR-10 INR split."
    python - "$TARGET" "$SPLIT_NOAUG" <<'PY'
import glob
import json
import os
import re
import sys
root, output = sys.argv[1:]
files = glob.glob(os.path.join(root, "randinit_smaller_*", "net*.pth"))
idx_re = re.compile(r"net(\d+)\.pth$")
lbl_re = re.compile(r"_(\d+)s")
base = {}
for path in files:
    dirname = os.path.basename(os.path.dirname(path))
    if "aug" in dirname:
        continue
    m_id = idx_re.search(os.path.basename(path))
    m_lbl = lbl_re.search(dirname)
    if m_id is None or m_lbl is None:
        continue
    idx = int(m_id.group(1))
    if idx in base:
        raise RuntimeError(f"Duplicate original INR for image {idx}: {path}")
    base[idx] = (os.path.relpath(path, root), int(m_lbl.group(1)))
if len(base) != 60000:
    raise RuntimeError(
        f"Expected 60,000 non-augmented CIFAR-10 INRs; got {len(base)}"
    )
result = {s: {"path": [], "label": []} for s in ("train", "val", "test")}
for idx, (path, label) in sorted(base.items()):
    split = "train" if idx < 45000 else ("val" if idx < 50000 else "test")
    result[split]["path"].append(path)
    result[split]["label"].append(label)
counts = [len(result[s]["path"]) for s in ("train", "val", "test")]
if counts != [45000, 5000, 10000]:
    raise RuntimeError(f"Unexpected non-augmented splits: {counts}")
with open(output, "w") as handle:
    json.dump(result, handle)
print(f"[nfn-split] wrote {output}: train/val/test={counts}")
PY
}

log "Ensuring target directory exists..."
mkdir -p "$TARGET"
log "Target directory ready."

log "Step 1: checking whether all 60,000 original INRs are present."
if have_original_checkpoints; then
    log "Original CIFAR-10 INRs already extracted; skipping download and extraction."
    build_splits
    finish "$TARGET"
    exit 0
fi
log "Original CIFAR-10 INRs are incomplete or missing."

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

log "Step 3: extracting only original (non-augmented) SIREN weights."
mkdir -p "$TARGET"

if ! extract_noaug_only "$ARCHIVE" "$TARGET"; then
    log "Selective extraction failed. Keeping any existing checkpoints intact."
    rm -f "$ARCHIVE" "${ARCHIVE}.part"
    log "Re-downloading the upstream full archive for a clean retry..."
    download_gdrive "$GDRIVE_FILE_ID" "$ARCHIVE"
    if ! extract_noaug_only "$ARCHIVE" "$TARGET"; then
        die "Selective CIFAR-10 original INR extraction failed twice."
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

log "Step 7: checking all original CIFAR-10 INRs are present."
have_original_checkpoints \
    || die "Selective extraction is incomplete; refusing to build partial splits."

log "Step 8: building the non-augmented split only."
build_splits
require_file "$SPLIT_NOAUG"

log "Step 9: cleaning up downloaded archive."
cleanup_archive "$ARCHIVE"

finish "$TARGET"
