#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# Run the pinned, unmodified Kahana et al. implementation.
# Default: 5 probe counts x 3 seeds; optional --dry-run / --probes / --seeds.
exec python scripts/ProbeGen/classification/run_fmnist_inr_official_probe_count.py "$@"
