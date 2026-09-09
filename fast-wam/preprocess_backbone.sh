#!/usr/bin/env bash
# preprocess_backbone.sh
#
# One-time preprocessing script: extracts the ActionDiT backbone from
# Wan2.2-TI2V-5B and saves it as a standalone .pt file for Fast-WAM inference.
#
# Run on a GPU node after running download_checkpoint.sh.
#
# Requirements:
#   - GPU available (recommended: H100 or A100; uses bfloat16)
#   - Wan2.2-TI2V-5B must be referenced in fastwam.yaml (model config) OR
#     HF_HOME must point to the directory where the model was downloaded.
#
# Usage:
#   sbatch --gpus=1 --partition=GPU-shared fast-wam/preprocess_backbone.sh
#   OR run interactively on a GPU node:
#   bash fast-wam/preprocess_backbone.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents"
FASTWAM_DIR="${REPO_ROOT}/fast-wam/FastWAM"
HF_CACHE="/ocean/projects/cis250185p/asingal/.hf_cache"
OUTPUT_DIR="${REPO_ROOT}/fast-wam/checkpoints"
OUTPUT_PT="${OUTPUT_DIR}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"

MODEL_CONFIG="${FASTWAM_DIR}/configs/model/fastwam.yaml"
PREPROCESS_SCRIPT="${FASTWAM_DIR}/scripts/preprocess_action_dit_backbone.py"
CONDA_ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/fastwam_env"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
echo "[preprocess_backbone] Activating fastwam_env ..."
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"

export HF_HOME="${HF_CACHE}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [ ! -f "${MODEL_CONFIG}" ]; then
    echo "[ERROR] Model config not found: ${MODEL_CONFIG}"
    exit 1
fi

if [ ! -f "${PREPROCESS_SCRIPT}" ]; then
    echo "[ERROR] Preprocessing script not found: ${PREPROCESS_SCRIPT}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Run preprocessing
# ---------------------------------------------------------------------------
echo "[preprocess_backbone] Running ActionDiT backbone extraction ..."
echo "  Model config : ${MODEL_CONFIG}"
echo "  Output file  : ${OUTPUT_PT}"
echo "  Device       : cuda"
echo "  Dtype        : bfloat16"
echo "  Alpha scaling: true"
echo ""

python "${PREPROCESS_SCRIPT}" \
    --model-config "${MODEL_CONFIG}" \
    --output       "${OUTPUT_PT}" \
    --device       cuda \
    --dtype        bfloat16 \
    --apply-alpha-scaling true

# ---------------------------------------------------------------------------
# Verify output
# ---------------------------------------------------------------------------
if [ ! -f "${OUTPUT_PT}" ]; then
    echo "[ERROR] Expected output file not found after preprocessing: ${OUTPUT_PT}"
    exit 1
fi

echo ""
echo "[preprocess_backbone] Done. Output file:"
ls -lh "${OUTPUT_PT}"
