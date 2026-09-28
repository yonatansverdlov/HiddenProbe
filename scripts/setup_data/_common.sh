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
    command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: $cmd"
}

require_file() {
    local file="$1"
    [[ -s "$file" ]] || die "Required file missing or empty: $file"
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

download_gdrive() {
    local file_id="$1"
    local dest="$2"
    local url="https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t"

    log "Downloading Google Drive file id=${file_id}"
    download_url "$url" "$dest"
}

download_url() {
    local url="$1"
    local dest="$2"
    local part="${dest}.part"

    mkdir -p "$(dirname "$dest")"

    if [[ -s "$dest" ]]; then
        log "Using existing downloaded file: $dest"
        return 0
    fi

    if command -v curl >/dev/null 2>&1; then
        if [[ -s "$part" ]]; then
            log "Resuming partial download: $part"
            if ! curl --fail --location --retry 5 --retry-delay 2 \
                --continue-at - --progress-bar --output "$part" "$url"; then
                log "Resume failed; restarting download from scratch."
                rm -f "$part"
                curl --fail --location --retry 5 --retry-delay 2 \
                    --progress-bar --output "$part" "$url"
            fi
        else
            log "Downloading: $url"
            curl --fail --location --retry 5 --retry-delay 2 \
                --progress-bar --output "$part" "$url"
        fi
    elif command -v wget >/dev/null 2>&1; then
        log "Downloading with wget: $url"
        wget --continue --tries=5 --timeout=30 \
            --progress=bar:force:noscroll -O "$part" "$url"
    else
        die "Neither curl nor wget is installed."
    fi

    [[ -s "$part" ]] || die "Download produced an empty file: $part"
    mv -f "$part" "$dest"
    log "Download complete: $dest"
}

extract_zip_with_progress() {
    local archive="$1"
    local dest="$2"
    local label="${3:-ZIP extraction}"

    require_cmd unzip

    local total
    total="$(unzip -Z1 "$archive" 2>/dev/null | wc -l | tr -d '[:space:]')"

    if [[ -z "$total" || "$total" -le 0 ]]; then
        log "Could not count ZIP entries; extracting normally."
        unzip -o "$archive" -d "$dest"
        return
    fi

    log "$label: $total entries"

    local awk_program='
        BEGIN { width=40; count=0; last=-1 }
        /(^|[[:space:]])(inflating:|extracting:|creating:|linking:)/ {
            count++
            pct=int((count*100)/total)
            if (pct>100) pct=100
            if (pct != last) {
                filled=int(width*pct/100)
                bar=""
                for (i=0; i<filled; i++) bar=bar "#"
                for (i=filled; i<width; i++) bar=bar "-"
                printf "\r[setup] %s: [%s] %3d%% (%d/%d)", label, bar, pct, count, total > "/dev/stderr"
                fflush("/dev/stderr")
                last=pct
            }
        }
        END { printf "\n" > "/dev/stderr"; fflush("/dev/stderr") }
    '

    unzip -o "$archive" -d "$dest" 2>&1 \
        | awk -v total="$total" -v label="$label" "$awk_program"
}

extract_tar_xz_with_progress() {
    local archive="$1"
    local dest="$2"
    local label="${3:-tar.xz extraction}"

    require_cmd tar

    local total
    total="$(tar -tJf "$archive" 2>/dev/null | wc -l | tr -d '[:space:]')"

    if [[ -z "$total" || "$total" -le 0 ]]; then
        log "Could not count tar entries; extracting normally."
        tar -xJf "$archive" -C "$dest"
        return
    fi

    log "$label: $total entries"

    local awk_program='
        BEGIN { width=40; count=0; last=-1 }
        {
            count++
            pct=int((count*100)/total)
            if (pct>100) pct=100
            if (pct != last) {
                filled=int(width*pct/100)
                bar=""
                for (i=0; i<filled; i++) bar=bar "#"
                for (i=filled; i<width; i++) bar=bar "-"
                printf "\r[setup] %s: [%s] %3d%% (%d/%d)", label, bar, pct, count, total > "/dev/stderr"
                fflush("/dev/stderr")
                last=pct
            }
        }
        END { printf "\n" > "/dev/stderr"; fflush("/dev/stderr") }
    '

    tar -xJvf "$archive" -C "$dest" 2>&1 \
        | awk -v total="$total" -v label="$label" "$awk_program"
}

cleanup_archive() {
    local archive="$1"

    if [[ "${KEEP_ARCHIVES:-0}" == "1" ]]; then
        log "KEEP_ARCHIVES=1; keeping archive: $archive"
    else
        rm -f "$archive" "${archive}.part"
        log "Removed archive: $archive"
    fi
}

finish() {
    local target="$1"
    log "Dataset ready at: $target"
}
