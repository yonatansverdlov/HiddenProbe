#!/usr/bin/env bash
set -euo pipefail

ARCHITECTURE="${ARCHITECTURE:?Set ARCHITECTURE=resnet18 or resnet50}"
PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/classification/modelj_cifar100_resnet}"
CIFAR_ROOT="${CIFAR_ROOT:-data/raw/cifar100}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
SPLITS="${SPLITS:-train val test}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

"$PYTHON" scripts/model_j_resnet/generate_modelj_resnet.py \
  --architecture "$ARCHITECTURE" \
  --output_root "$OUTPUT_ROOT" \
  --cifar_root "$CIFAR_ROOT" \
  --device "$DEVICE" \
  --num_workers "$NUM_WORKERS" \
  --num_shards "$NUM_SHARDS" \
  --shard_id "$SHARD_ID" \
  --splits $SPLITS \
  $EXTRA_ARGS \
  "$@"
