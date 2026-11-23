#!/usr/bin/env bash
# ==============================================================================
# Flood-AI: Dataset Retrieval & Directory Setup Script
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"

echo "Initializing data directories..."
mkdir -p "${RAW_DIR}/sen1floods11"
mkdir -p "${RAW_DIR}/sensors"
mkdir -p "${RAW_DIR}/weather"
mkdir -p "${ROOT_DIR}/data/processed"
mkdir -p "${ROOT_DIR}/data/external"

echo "Fetching Sen1Floods11 benchmark metadata splits..."
SPLITS_URL="https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/splits/flood_handlabeled"

for split in train val test; do
    echo "Downloading ${split} split CSV..."
    curl -sSL "${SPLITS_URL}/flood_${split}_data.csv" -o "${RAW_DIR}/sen1floods11/flood_${split}_data.csv" || true
done

echo "Setup complete. Raw data root: ${RAW_DIR}"
