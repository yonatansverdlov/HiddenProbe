#!/usr/bin/env bash
# Threshold-specific ProbeGen baseline matching the HiddenProbe experiment layout.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUT=0.8 bash "$SCRIPT_DIR/run_agnews_transformer.sh" "$@"
