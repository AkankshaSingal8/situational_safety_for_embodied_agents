#!/usr/bin/env bash
# Sets up the qwen conda env for Qwen2-VL inference.
# NOTE: As of 2026-04-20 this env is already installed at
#       /ocean/projects/cis250185p/asingal/envs/qwen
#       with transformers==5.5.0.dev0, torch==2.7.1+cu118, qwen-vl-utils==0.0.14
#
# Run ONCE on a GPU node if recreating from scratch.
# Usage: bash qwen_vlm_env_setup.sh
set -euo pipefail

ENV_NAME="qwen"
ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/${ENV_NAME}"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Creating qwen VLM env at ${ENV_PREFIX}"
conda create -p "${ENV_PREFIX}" python=3.10 -y
conda activate "${ENV_PREFIX}"

pip install --upgrade pip

# PyTorch — same CUDA version as openvla_libero_merged for driver compatibility
pip install \
  torch==2.2.0 \
  torchvision==0.17.0 \
  torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu118

# Standard transformers (>= 4.45.0 for Qwen2-VL support)
# NOTE: Do NOT install moojink fork here — it breaks Qwen2-VL
pip install \
  "transformers>=4.45.0" \
  accelerate \
  qwen-vl-utils \
  bitsandbytes \
  Pillow \
  scipy \
  numpy \
  sentencepiece

echo "qwen env ready at ${ENV_PREFIX}"
echo "Verify: conda run -n qwen python -c 'from transformers import Qwen2VLForConditionalGeneration; print(\"OK\")'"
