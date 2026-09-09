#!/usr/bin/env bash
# Creates dreamzero_env for running the DreamZero WAM server.
# Tested on Bridges2 HPC with H100 GPUs and CUDA 12.9.
#
# Prerequisites:
#   - CUDA 12.9 module loaded (module load cuda/12.9 or similar)
#   - Run on a GPU node (flash-attn requires CUDA headers)
#
# Usage: bash dream-policy/setup_dreamzero_env.sh
#
# If CUDA 12.9 is unavailable, change cu129 → cu124 in the torch install line.

set -euo pipefail

ENV_PATH=/ocean/projects/cis250185p/asingal/envs/dreamzero_env
REPO_ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents

echo "=== Creating conda env at $ENV_PATH (Python 3.11) ==="
conda create -p "$ENV_PATH" python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

echo "=== Installing PyTorch 2.8.0 + CUDA 12.9 ==="
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu129
# Fallback if cu129 wheel unavailable:
# pip install torch==2.8.0 torchvision \
#     --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing DreamZero core dependencies ==="
pip install \
    "transformers==4.51.3" \
    "diffusers==0.30.2" \
    "peft==0.5.0" \
    "accelerate>=1.0.0" \
    "safetensors>=0.5.0" \
    "einops>=0.8.0" \
    "hydra-core>=1.3.0" \
    "deepspeed>=0.16.0" \
    "flask>=3.0.0" \
    "flask-socketio>=5.5.0" \
    "python-socketio>=5.13.0" \
    "huggingface-hub>=0.36.0" \
    "numpy>=1.26.4" \
    "pillow>=11.0.0" \
    "opencv-python-headless>=4.11.0" \
    "imageio>=2.37.0" \
    "tqdm>=4.67.0"

echo "=== Installing DreamZero repo (editable) ==="
if [ -f "$REPO_ROOT/dream-policy/dreamzero/setup.py" ] || \
   [ -f "$REPO_ROOT/dream-policy/dreamzero/pyproject.toml" ]; then
    pip install -e "$REPO_ROOT/dream-policy/dreamzero"
elif [ -f "$REPO_ROOT/dream-policy/dreamzero/requirements.txt" ]; then
    pip install -r "$REPO_ROOT/dream-policy/dreamzero/requirements.txt"
else
    echo "WARNING: No installable package found in dream-policy/dreamzero/."
    echo "         Install DreamZero deps manually after cloning the repo."
fi

echo "=== Installing flash-attn (requires CUDA headers; takes 5-15 min) ==="
pip install flash-attn --no-build-isolation

echo "=== Verifying GPU access ==="
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
print('CUDA version:', torch.version.cuda)
print('PyTorch version:', torch.__version__)
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate $ENV_PATH"
