#!/usr/bin/env bash
set -euo pipefail
export ARCHITECTURE=resnet50
exec bash scripts/model_j_resnet/run_modelj_resnet.sh "$@"
