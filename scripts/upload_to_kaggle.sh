#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${REPO_ROOT}/data/generated/kaggle-20260903-01"
KERNEL_DIR="${REPO_ROOT}/notebooks/kaggle"

echo "=== 1. Uploading dataset to Kaggle ==="
cd "${DATASET_DIR}"
kaggle datasets create -p . -r zip || kaggle datasets version -p . -m "Updated dataset and source" -r zip

echo "=== 2. Pushing training kernel to Kaggle GPU ==="
cd "${REPO_ROOT}"
kaggle kernels push -p "${KERNEL_DIR}"

echo "=== 3. Checking kernel status ==="
kaggle kernels status trietp1253201581/anomaly-representation-training-gpu
