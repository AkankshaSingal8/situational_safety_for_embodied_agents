#!/bin/bash
set -euo pipefail

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# --- 1. Create Python 3.8 env ---
conda create -n libero_env python=3.8.13 -y
conda activate libero_env

echo "=== Python: $(python --version) ==="

pip install --upgrade pip

# --- 2. Install PyTorch cu113 (as pinned in requirements.txt) ---
# cu113 is the only CUDA variant with torch 1.11 wheels
pip install \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# --- 3. Install av via conda-forge (avoids ffmpeg build error) ---
conda install -c conda-forge "av>=9.0.0" -y --freeze-installed

# --- 4. Install safelibero/libero editable ---
pip install -e /ocean/projects/cis250185p/asingal/SafeLIBERO/safelibero --no-deps

# --- 5. Install all requirements from main/requirements.txt
#        EXCEPT torch/torchvision/torchaudio (already installed)
#        and openpi-client (handled separately) ---
pip install \
    absl-py==2.1.0 \
    addict==2.4.0 \
    bddl==1.0.1 \
    clarabel==0.9.0 \
    cloudpickle==2.1.0 \
    configargparse==1.7.1 \
    cvxpy==1.5.2 \
    dm-tree==0.1.8 \
    easydict==1.9 \
    einops==0.4.1 \
    flask==3.0.3 \
    glfw==1.12.0 \
    groundingdino-py==0.4.0 \
    gym==0.25.2 \
    h5py==3.11.0 \
    huggingface-hub==0.36.0 \
    hydra-core==1.2.0 \
    imageio==2.35.1 \
    imageio-ffmpeg==0.5.1 \
    joblib==1.4.2 \
    llvmlite==0.36.0 \
    matplotlib==3.5.3 \
    msgpack==1.1.1 \
    mujoco==3.2.3 \
    numba==0.53.1 \
    "numpy==1.22.4" \
    "open3d==0.18.0" \
    opencv-python==4.6.0.66 \
    osqp==1.0.5 \
    pandas==2.0.3 \
    pillow==10.4.0 \
    plotly==6.5.0 \
    pycocotools==2.0.7 \
    pydantic==2.10.6 \
    pyopengl==3.1.7 \
    pyquaternion==0.9.9 \
    rich==13.9.4 \
    robomimic==0.2.0 \
    robosuite==1.4.1 \
    safetensors==0.5.3 \
    scikit-learn==1.3.2 \
    scipy==1.10.1 \
    supervision==0.6.0 \
    timm==1.0.22 \
    tokenizers==0.12.1 \
    tqdm==4.67.1 \
    transformers==4.21.1 \
    tyro==0.9.2 \
    wandb==0.13.1 \
    "websockets==13.1" \
    pyyaml==6.0.2 \
    svgwrite==1.4.3 \
    pyzmq \
    pynput==1.7.7

# Install zhipuai and zai-sdk separately bypassing pyjwt conflict
pip install --no-deps zhipuai==2.1.5.20250825
pip install --no-deps zai-sdk==0.0.4.2

# Install a pyjwt version that works for both at runtime (2.8.0 satisfies zhipuai)
pip install "pyjwt==2.8.0"

# --- 6. Install openpi-client (connects to serve_policy server) ---
pip install -e /ocean/projects/cis250185p/asingal/vlsa-aegis/openpi/packages/openpi-client --no-deps

pip install "cachetools>=4.2.2" "httpx>=0.23.0"
pip install httpx future
conda install -c conda-forge libstdcxx-ng -y
conda install -c conda-forge "pillow=10.4.0" -y

# --- 7. Verify critical imports ---
python -c "
import robosuite; print('robosuite: OK')
from libero.libero.envs import OffScreenRenderEnv; print('libero envs: OK')
import openpi_client; print('openpi_client: OK')
import zhipuai; print('zhipuai: OK')
import groundingdino; print('groundingdino: OK')

"
python -c "from PIL import Image; print('PIL: OK')"
echo ""
echo "=== libero_env setup complete ==="