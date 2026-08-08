#!/usr/bin/env bash
# Downloads and extracts the dataset, metadata, splits, and trained checkpoint
# published on Zenodo (https://doi.org/10.5281/zenodo.21765014) into data/.
#
# Safe to re-run: already-downloaded/extracted files are skipped.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${REPO_ROOT}/data"
ZENODO_BASE="https://zenodo.org/records/21765014/files"

mkdir -p "${DATA_DIR}"

if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl is required but not installed." >&2
    exit 1
fi

declare -A FRESHLY_DOWNLOADED

download() {
    local filename="$1"
    local dest="${DATA_DIR}/${filename}"
    if [[ -f "${dest}" ]]; then
        echo "Skipping download, already exists: ${filename}"
        return
    fi
    echo "Downloading ${filename}..."
    curl -L --fail --progress-bar -o "${dest}" "${ZENODO_BASE}/${filename}?download=1"
    FRESHLY_DOWNLOADED["${filename}"]=1
}

# Only removes the archive if this run downloaded it; a pre-existing archive
# (e.g. kept intentionally as a backup) is never deleted.
# check_path (relative to DATA_DIR) is used for the idempotency check; dest_dir
# is where the archive is actually extracted (defaults to DATA_DIR itself).
extract_and_cleanup() {
    local archive="$1"
    local check_path="$2"
    local dest_dir="${3:-${DATA_DIR}}"
    if [[ -d "${DATA_DIR}/${check_path}" ]]; then
        echo "Skipping extraction, already exists: ${check_path}/"
        return
    fi
    echo "Extracting ${archive}..."
    mkdir -p "${dest_dir}"
    tar -xzf "${DATA_DIR}/${archive}" -C "${dest_dir}"
    if [[ "${FRESHLY_DOWNLOADED[${archive}]:-0}" == "1" ]]; then
        rm -f "${DATA_DIR}/${archive}"
    fi
}

# Dataset splits, hierarchy tree, and label maps (~2 MB)
download "2026-01-19_portable.tar.gz"
extract_and_cleanup "2026-01-19_portable.tar.gz" "2026-01-19_portable"

# Supplementary class list (reference only, not read by the code)
download "ClassificationClasses_IHC.csv"

# Insect image dataset (~4 GB compressed, ~7 GB extracted). The archive's
# top-level folder is Crops_clean/, not IHC_dataset/, so it must be extracted
# into an IHC_dataset/ subfolder to match config.py's DATASET_ROOT.
download "IHC_dataset.tar.gz"
extract_and_cleanup "IHC_dataset.tar.gz" "IHC_dataset/Crops_clean" "${DATA_DIR}/IHC_dataset"

# DuckDB metadata database (~40 MB)
download "insect_images_public.duckdb"

# Trained model checkpoint (~90 MB)
download "model_Insect_100_96_bz1024_resnet18_OneCycle_2026-01-19_01-09.pth"

echo "Done. Data is available in ${DATA_DIR}/"
