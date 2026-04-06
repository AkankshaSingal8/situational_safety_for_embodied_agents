#!/usr/bin/env bash
# vlm_env_setup.sh
#
# Creates two conda environments on Bridges-2:
#   1) openvla  – OpenVLA-OFT policy + LIBERO/MuJoCo evaluation
#   2) qwen    – Qwen2.5-VL / Qwen3-VL inference (transformers ≥ 4.45)
#
# Prerequisites:
#   - module load anaconda3 (or conda already on PATH)
#   - CUDA drivers available (H100/L40S nodes)
#
# Usage:
#   bash vlm_env_setup.sh          # create both envs
#   bash vlm_env_setup.sh openvla  # create only the openvla env
#   bash vlm_env_setup.sh qwen     # create only the qwen env

'''
useful links:
https://huggingface.co/datasets/THURCSCT/SafeLIBERO
https://github.com/openvla/openvla

'''
set -euo pipefail

TARGET="${1:-all}"

# ─────────────────────────────────────────────────────────────────────────────
# openvla env – OpenVLA-OFT + LIBERO + MuJoCo + SafeLIBERO evaluation
# ─────────────────────────────────────────────────────────────────────────────
create_openvla_env() {
    echo "============================================================"
    echo "  Creating 'openvla' conda environment"
    echo "============================================================"

    conda create -y -n openvla python=3.10

    eval "$(conda shell.bash hook)"
    conda activate openvla

    # PyTorch (CUDA 12.x wheels for Bridges-2 H100 / L40S)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    # LIBERO + robosuite (MuJoCo simulation)
    pip install libero robosuite mujoco

    # OpenVLA-OFT dependencies
    pip install transformers accelerate safetensors timm
    pip install prismatic-vlms
    pip install draccus wandb

    # General utilities
    pip install numpy scipy pillow tqdm

    # EGL rendering for headless MuJoCo on HPC
    pip install mujoco-python-viewer PyOpenGL PyOpenGL-accelerate

    conda deactivate
    echo ">>> 'openvla' environment created successfully."
}

# ─────────────────────────────────────────────────────────────────────────────
# qwen env – Qwen VLM inference (called via subprocess from the main env)
# ─────────────────────────────────────────────────────────────────────────────
create_qwen_env() {
    echo "============================================================"
    echo "  Creating 'qwen' conda environment"
    echo "============================================================"

    conda create -y -n qwen python=3.10

    eval "$(conda shell.bash hook)"
    conda activate qwen

    # PyTorch (CUDA 12.x)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    # Transformers (≥4.45 required for Qwen2.5-VL / Qwen3-VL)
    pip install "transformers>=4.45" accelerate safetensors

    # 4-bit quantisation support (optional but recommended for 7B+ models)
    pip install bitsandbytes

    # Image handling
    pip install pillow

    # Flash-attention 2 (speeds up multi-image / video inference)
    pip install flash-attn --no-build-isolation

    # Qwen-VL image/video processing helpers
    pip install qwen-vl-utils

    # General utilities
    pip install numpy tqdm

    conda deactivate
    echo ">>> 'qwen' environment created successfully."
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────
case "$TARGET" in
    openvla)
        create_openvla_env
        ;;
    qwen)
        create_qwen_env
        ;;
    all)
        create_openvla_env
        create_qwen_env
        ;;
    *)
        echo "Usage: bash vlm_env_setup.sh [openvla|qwen|all]"
        exit 1
        ;;
esac

echo ""
echo "Done. Activate with:"
echo "  conda activate openvla   # for LIBERO + OpenVLA-OFT evaluation"
echo "  conda activate qwen      # for Qwen VLM inference"
