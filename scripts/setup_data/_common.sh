#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$DATA_ROOT/.downloads}"

mkdir -p "$DOWNLOAD_DIR"

log() {
    echo "[setup] $*"
}

die() {
    echo "[setup] ERROR: $*" >&2
    exit 1
}

require_cmd() {
    local cmd="$1"
    log "Checking required command: $cmd"
    command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: $cmd"
    log "Command available: $cmd"
}

require_file() {
    local file="$1"
    log "Checking required file: $file"
    [[ -s "$file" ]] || die "Required file missing or empty: $file"
    log "File OK: $file"
}

require_dir_nonempty() {
    local dir="$1"
    log "Checking required directory: $dir"
    [[ -d "$dir" ]] || die "Required directory not found: $dir"
    find "$dir" -mindepth 1 -print -quit 2>/dev/null | grep -q . \
        || die "Required directory is empty: $dir"
    log "Directory OK: $dir"
}

dataset_files_ready() {
    local name="$1"
    local target="$2"
    shift 2

    log "Checking whether $name dataset is already fully installed..."

    if [[ ! -d "$target" ]]; then
        log "$name target directory does not exist yet: $target"
        return 1
    fi

    local req
    for req in "$@"; do
        if [[ ! -s "$target/$req" ]]; then
            log "$name is not ready yet; missing or empty: $target/$req"
            return 1
        fi
        log "Found required dataset file: $target/$req"
    done

    log "$name dataset already exists at: $target"
    log "Skipping download and extraction."
    return 0
}

# ---------------------------------------------------------------------------
# Downloads with progress bars
# ---------------------------------------------------------------------------
download_url() {
    local url="$1"
    local dest="$2"
    local part="${dest}.part"

    log "Preparing download destination: $dest"
    mkdir -p "$(dirname "$dest")"

    if [[ -s "$dest" ]]; then
        log "Using existing archive/file: $dest"
        return 0
    fi

    if command -v curl >/dev/null 2>&1; then
        if [[ -s "$part" ]]; then
            log "Found partial download: $part"
            log "Attempting to resume download with progress bar..."

            if ! curl \
                --fail \
                --location \
                --retry 5 \
                --retry-delay 2 \
                --continue-at - \
                --progress-bar \
                --output "$part" \
                "$url"; then

                log "Resume failed."
                log "Deleting partial download and restarting from scratch."
                rm -f "$part"

                log "Starting fresh download with progress bar..."
                curl \
                    --fail \
                    --location \
                    --retry 5 \
                    --retry-delay 2 \
                    --progress-bar \
                    --output "$part" \
                    "$url"
            fi
        else
            log "Starting download with progress bar..."
            log "Source: $url"

            curl \
                --fail \
                --location \
                --retry 5 \
                --retry-delay 2 \
                --progress-bar \
                --output "$part" \
                "$url"
        fi

    elif command -v wget >/dev/null 2>&1; then
        if [[ -s "$part" ]]; then
            log "Found partial download: $part"
            log "Resuming download with progress bar..."
        else
            log "Starting download with progress bar..."
            log "Source: $url"
        fi

        wget \
            --continue \
            --tries=5 \
            --timeout=30 \
            --progress=bar:force:noscroll \
            -O "$part" \
            "$url"
    else
        die "Neither curl nor wget is installed."
    fi

    log "Checking downloaded file is non-empty..."
    [[ -s "$part" ]] || die "Download produced an empty file: $part"

    log "Download finished successfully."
    log "Promoting partial file to final filename..."
    mv -f "$part" "$dest"
    log "Saved completed file: $dest"
}

download_gdrive() {
    local file_id="$1"
    local dest="$2"
    local part="${dest}.part"

    log "Preparing Google Drive download destination: $dest"
    mkdir -p "$(dirname "$dest")"

    if [[ -s "$dest" ]]; then
        log "Using existing archive/file: $dest"
        return 0
    fi

    require_cmd python

    log "Checking Python package: gdown"
    if ! python -c 'import gdown' >/dev/null 2>&1; then
        log "gdown is not installed."
        log "Installing gdown in the active Python environment..."
        python -m pip install gdown
        log "gdown installation complete."
    else
        log "gdown is available."
    fi

    if [[ -s "$part" ]]; then
        log "Found partial Google Drive download: $part"
        log "Attempting to resume it with progress bar."
    else
        log "Starting Google Drive download with progress bar."
    fi

    log "Google Drive file id: $file_id"

    python - "$file_id" "$part" <<'PY_GDOWN'
import sys
import gdown

file_id, output = sys.argv[1], sys.argv[2]
url = f"https://drive.google.com/uc?id={file_id}"

result = gdown.download(
    url=url,
    output=output,
    quiet=False,
    resume=True,
)

if result is None:
    raise SystemExit("gdown download failed")
PY_GDOWN

    log "Checking downloaded file is non-empty..."
    [[ -s "$part" ]] || die "Google Drive download produced an empty file: $part"

    log "Google Drive download finished successfully."
    log "Promoting partial file to final filename..."
    mv -f "$part" "$dest"
    log "Saved completed file: $dest"
}

# ---------------------------------------------------------------------------
# Extraction progress helpers
#
# tar/tar.xz is streamed exactly once. Python reports bytes read while tar
# extracts from stdin.
#
# ZIP extraction counts central-directory entries first (fast metadata read),
# then performs the actual extraction once while displaying file-count progress.
# ---------------------------------------------------------------------------

_stream_archive_with_progress() {
    local archive="$1"
    local label="${2:-Extraction}"

    require_cmd python

    python - "$archive" "$label" <<'PY_PROGRESS'
import os
import sys
import time

path = sys.argv[1]
label = sys.argv[2]

total = os.path.getsize(path)
chunk_size = 4 * 1024 * 1024
width = 40

done = 0
start = time.time()
last_pct = -1

with open(path, "rb") as src:
    out = sys.stdout.buffer

    while True:
        chunk = src.read(chunk_size)
        if not chunk:
            break

        try:
            out.write(chunk)
        except BrokenPipeError:
            raise SystemExit(1)

        done += len(chunk)
        pct = int(done * 100 / total) if total else 100

        if pct != last_pct:
            elapsed = max(time.time() - start, 1e-6)
            speed = done / elapsed
            remaining = max(total - done, 0)
            eta = remaining / speed if speed > 0 else 0

            filled = int(width * pct / 100)
            bar = "#" * filled + "-" * (width - filled)

            done_mib = done / (1024 ** 2)
            total_mib = total / (1024 ** 2)
            speed_mib = speed / (1024 ** 2)

            sys.stderr.write(
                f"\r[setup] {label}: [{bar}] {pct:3d}% "
                f"{done_mib:,.0f}/{total_mib:,.0f} MiB "
                f"{speed_mib:,.1f} MiB/s ETA {eta:,.0f}s"
            )
            sys.stderr.flush()
            last_pct = pct

    out.flush()

sys.stderr.write("\n")
sys.stderr.flush()
PY_PROGRESS
}

extract_tar_xz_with_progress() {
    local archive="$1"
    local dest="$2"
    local label="${3:-tar.xz extraction}"

    log "Starting extraction with progress bar..."

    if _stream_archive_with_progress "$archive" "$label" \
        | tar -xJf - -C "$dest"; then
        log "Extraction progress: 100%"
        return 0
    fi

    echo >&2
    log "Extraction command failed."
    return 1
}

extract_tar_with_progress() {
    local archive="$1"
    local dest="$2"
    local label="${3:-tar extraction}"

    log "Starting extraction with progress bar..."

    if _stream_archive_with_progress "$archive" "$label" \
        | tar -xf - -C "$dest"; then
        log "Extraction progress: 100%"
        return 0
    fi

    echo >&2
    log "Extraction command failed."
    return 1
}

extract_zip_with_progress() {
    local archive="$1"
    local dest="$2"
    local label="${3:-ZIP extraction}"

    require_cmd unzip

    log "Counting ZIP entries for extraction progress..."

    local total
    total="$(unzip -Z1 "$archive" 2>/dev/null | wc -l | tr -d '[:space:]')"

    if [[ -z "$total" || "$total" -le 0 ]]; then
        log "Could not determine ZIP entry count."
        log "Falling back to normal extraction output."
        unzip -o "$archive" -d "$dest"
        return $?
    fi

    log "ZIP contains $total entries."
    log "Starting extraction with progress bar..."

    local awk_program='
        BEGIN {
            width = 40
            count = 0
            last_pct = -1
        }

        /(^|[[:space:]])(inflating:|extracting:|creating:|linking:)/ {
            count++
            pct = int((count * 100) / total)
            if (pct > 100) pct = 100

            if (pct != last_pct) {
                filled = int(width * pct / 100)
                bar = ""

                for (i = 0; i < filled; i++) {
                    bar = bar "#"
                }

                for (i = filled; i < width; i++) {
                    bar = bar "-"
                }

                printf "\r[setup] %s: [%s] %3d%% (%d/%d entries)", \
                    label, bar, pct, count, total > "/dev/stderr"
                fflush("/dev/stderr")
                last_pct = pct
            }
        }

        END {
            printf "\n" > "/dev/stderr"
            fflush("/dev/stderr")
        }
    '

    if command -v stdbuf >/dev/null 2>&1; then
        if stdbuf -oL -eL unzip -o "$archive" -d "$dest" 2>&1 \
            | awk -v total="$total" -v label="$label" "$awk_program"; then
            log "Extraction progress: 100%"
            return 0
        fi
    else
        if unzip -o "$archive" -d "$dest" 2>&1 \
            | awk -v total="$total" -v label="$label" "$awk_program"; then
            log "Extraction progress: 100%"
            return 0
        fi
    fi

    log "Extraction command failed."
    return 1
}

cleanup_stale_extract_dirs() {
    local parent="$1"
    local prefix="$2"

    log "Checking for stale partial extraction directories..."

    shopt -s nullglob
    local dirs=("$parent"/"$prefix"*)
    shopt -u nullglob

    if (( ${#dirs[@]} > 0 )); then
        log "Found ${#dirs[@]} stale extraction directory/directories."
        log "Removing stale partial extraction data..."
        rm -rf -- "${dirs[@]}"
        log "Stale extraction data removed."
    else
        log "No stale extraction directories found."
    fi
}

cleanup_archive() {
    local archive="$1"

    if [[ "${KEEP_ARCHIVES:-0}" == "1" ]]; then
        log "KEEP_ARCHIVES=1; keeping archive: $archive"
    else
        log "Removing downloaded archive to save disk space..."
        rm -f "$archive" "${archive}.part"
        log "Archive removed."
    fi
}

finish() {
    local target="$1"
    log "Setup completed successfully."
    log "Dataset ready at: $target"
}

# ---------------------------------------------------------------------------
# Small Transformer Zoo helpers (transformer accuracy prediction; used by regression_*_transformer.sh)
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-python}"

# Small Transformer Zoo on the HuggingFace Hub (one zip archive per zoo).
HF_REPO="anonymized-acamedia/Small-Transformer-Zoo"
HF_BASE_URL="https://huggingface.co/datasets/${HF_REPO}/resolve/main"

# The zoo is installed when its wrapper dir holds at least one epoch-75 checkpoint (<run_id>_75_<acc>.pt).
zoo_ready() {
    local name="$1"
    local target="$2"

    log "Checking whether the $name zoo is already installed under: $target"
    if [[ -d "$target" ]] && find "$target" -type f -name '*_75_*.pt' -print -quit 2>/dev/null | grep -q .; then
        log "$name zoo already exists at: $target"
        return 0
    fi
    log "$name zoo is not installed yet."
    return 1
}

# The consolidated cache (`python main.py transformer cache`) = <DATA_ROOT>/tpcache_<dataset>_{train,val,test}_s<seed>.pt
cache_ready() {
    local dataset="$1"    # mnist | agnews
    local seed="${2:-0}"

    log "Checking whether the $dataset consolidated cache exists (split seed $seed)..."
    local split
    for split in train val test; do
        if [[ ! -s "$DATA_ROOT/tpcache_${dataset}_${split}_s${seed}.pt" ]]; then
            log "$dataset cache not complete yet (missing split: $split)."
            return 1
        fi
    done
    log "$dataset cache found."
    return 0
}

# install_zoo NAME ARCHIVE_NAME TARGET
#   Downloads <HF_BASE_URL>/<ARCHIVE_NAME> (resumable), extracts it into a temporary directory next to TARGET,
#   and moves the extracted tree to TARGET. A failed extraction deletes the archive and retries once.
install_zoo() {
    local name="$1"
    local archive_name="$2"
    local target="$3"
    local archive="$DOWNLOAD_DIR/$archive_name"
    local url="$HF_BASE_URL/$archive_name"

    require_cmd unzip

    local parent
    parent="$(dirname "$target")"
    mkdir -p "$parent"
    local tmp_prefix=".$(basename "$target")_extract."
    cleanup_stale_extract_dirs "$parent" "$tmp_prefix"

    local attempt
    for attempt in 1 2; do
        if [[ -s "$archive" ]]; then
            log "Using existing archive: $archive"
        else
            log "Downloading the $name zoo from the HuggingFace Hub (large archive; resumable)..."
            download_url "$url" "$archive"
        fi

        local tmp
        tmp="$(mktemp -d "$parent/${tmp_prefix}XXXXXX")"
        log "Extracting $archive_name into $tmp ..."
        if unzip -q "$archive" -d "$tmp"; then
            # Install the extracted tree under TARGET (main.py globs recursively beneath it).
            rm -rf "$target"
            mkdir -p "$target"
            mv "$tmp"/* "$target"/
            rm -rf "$tmp"
            log "$name zoo installed at: $target"
            return 0
        fi

        log "Extraction failed; treating the archive as corrupted or incomplete."
        rm -rf "$tmp"
        rm -f "$archive" "${archive}.part"
        [[ "$attempt" == 1 ]] && log "Re-downloading and retrying extraction once..."
    done
    die "$name extraction failed twice."
}

finish() {
    local target="$1"
    log "Setup completed successfully."
    log "Dataset ready at: $target"
}
