#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS=(
    "classification_cifar10.sh"
    "regression_mnist.sh"
    "regression_fmnist.sh"
    "regression_svhn.sh"
    "regression_cifar10_gs.sh"
    "regression_cifar10_wp.sh"
    "regression_mnist_transformer.sh"
    "regression_agnews_transformer.sh"
)

TOTAL="${#DATASETS[@]}"

echo "[setup-all] ============================================================"
echo "[setup-all] Starting setup for all $TOTAL datasets"
echo "[setup-all] Script directory: $SCRIPT_DIR"
echo "[setup-all] ============================================================"

for i in "${!DATASETS[@]}"; do
    SCRIPT_NAME="${DATASETS[$i]}"
    SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"
    NUM=$((i + 1))

    echo
    echo "[setup-all] ============================================================"
    echo "[setup-all] Dataset $NUM/$TOTAL"
    echo "[setup-all] Running: $SCRIPT_NAME"
    echo "[setup-all] ============================================================"

    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo "[setup-all] ERROR: script not found: $SCRIPT_PATH" >&2
        exit 1
    fi

    if bash "$SCRIPT_PATH"; then
        echo "[setup-all] Completed successfully: $SCRIPT_NAME"
    else
        STATUS=$?
        echo "[setup-all] FAILED: $SCRIPT_NAME (exit code $STATUS); stopping." >&2
        exit "$STATUS"
    fi
done

echo
echo "[setup-all] All $TOTAL dataset setups completed successfully."
