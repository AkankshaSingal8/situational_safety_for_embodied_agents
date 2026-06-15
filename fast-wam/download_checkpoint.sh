#!/usr/bin/env bash
# download_checkpoint.sh — Download Fast-WAM checkpoints for SafeLIBERO evaluation
#
# NOTE: This script can run on a LOGIN NODE — no GPU needed.
#       Expected runtime: ~1–2 hours depending on network speed.
#
# Downloads:
#   1. Wan-AI/Wan2.2-TI2V-5B (~22 GB) → HF cache (base model for ActionDiT backbone)
#   2. yuanty/fastwam:
#        libero_uncond_2cam224.pt              → fast-wam/checkpoints/
#        libero_uncond_2cam224_dataset_stats.json → fast-wam/checkpoints/

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HF_TOKEN_FILE="/jet/home/asingal/.cache/huggingface/token"
export HF_HOME="/ocean/projects/cis250185p/asingal/.hf_cache"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"

CONDA_ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/fastwam_env"

# ---------------------------------------------------------------------------
# HF token (optional — public repos work without one, but rate-limits differ)
# ---------------------------------------------------------------------------
if [[ -f "${HF_TOKEN_FILE}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$(cat "${HF_TOKEN_FILE}")"
    echo "[INFO] Loaded HF token from ${HF_TOKEN_FILE}"
else
    echo "[WARN] HF token file not found at ${HF_TOKEN_FILE} — proceeding without authentication"
fi

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
# shellcheck disable=SC1091
if [[ -f "/jet/home/asingal/.bashrc" ]]; then
    source "/jet/home/asingal/.bashrc" 2>/dev/null || true
fi

# Try conda.sh from common locations
for CONDA_SH in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/jet/home/asingal/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${CONDA_SH}" ]]; then
        source "${CONDA_SH}"
        break
    fi
done

if conda activate "${CONDA_ENV_PREFIX}" 2>/dev/null; then
    echo "[INFO] Activated conda env: ${CONDA_ENV_PREFIX}"
else
    echo "[WARN] Could not activate conda env ${CONDA_ENV_PREFIX} — using system Python"
    echo "       huggingface_hub must be installed in the active environment"
fi

# ---------------------------------------------------------------------------
# Verify huggingface-cli is available
# ---------------------------------------------------------------------------
if ! command -v huggingface-cli &>/dev/null; then
    echo "[ERROR] huggingface-cli not found. Install with: pip install huggingface_hub"
    exit 1
fi

# ---------------------------------------------------------------------------
# Create checkpoint directory
# ---------------------------------------------------------------------------
mkdir -p "${CHECKPOINT_DIR}"
echo "[INFO] Checkpoint directory: ${CHECKPOINT_DIR}"

# ---------------------------------------------------------------------------
# 1. Download Wan-AI/Wan2.2-TI2V-5B (~22 GB) into HF cache
# ---------------------------------------------------------------------------
WAN_CACHE_DIR="${HF_HOME}/hub/Wan-AI/Wan2.2-TI2V-5B"
echo ""
echo "=================================================================="
echo "[STEP 1/2] Downloading Wan-AI/Wan2.2-TI2V-5B (~22 GB)"
echo "           Destination: ${WAN_CACHE_DIR}"
echo "           This may take 60–90 minutes on a typical HPC network."
echo "=================================================================="

huggingface-cli download \
    Wan-AI/Wan2.2-TI2V-5B \
    --cache-dir "${HF_HOME}/hub" \
    --local-dir "${WAN_CACHE_DIR}"

echo ""
echo "[INFO] Wan2.2-TI2V-5B download complete."
du -sh "${WAN_CACHE_DIR}"

# ---------------------------------------------------------------------------
# 2. Download yuanty/fastwam checkpoint files
# ---------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "[STEP 2/2] Downloading yuanty/fastwam checkpoint files"
echo "           Destination: ${CHECKPOINT_DIR}"
echo "=================================================================="

huggingface-cli download \
    yuanty/fastwam \
    libero_uncond_2cam224.pt \
    libero_uncond_2cam224_dataset_stats.json \
    --local-dir "${CHECKPOINT_DIR}"

echo ""
echo "[INFO] Fast-WAM checkpoint download complete."
du -sh "${CHECKPOINT_DIR}/libero_uncond_2cam224.pt" 2>/dev/null || true
du -sh "${CHECKPOINT_DIR}/libero_uncond_2cam224_dataset_stats.json" 2>/dev/null || true
du -sh "${CHECKPOINT_DIR}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "All downloads complete."
echo ""
echo "  Wan2.2-TI2V-5B  : ${WAN_CACHE_DIR}"
echo "  LIBERO checkpoint: ${CHECKPOINT_DIR}/libero_uncond_2cam224.pt"
echo "  Dataset stats    : ${CHECKPOINT_DIR}/libero_uncond_2cam224_dataset_stats.json"
echo "=================================================================="
