#!/bin/bash
set -euo pipefail

echo "=== Using Python: $(which python) ==="
echo "=== Python version: $(python --version) ==="

module load cuda/12.6.1

# --- 1. Upgrade pip to get the newer resolver ---
pip install --upgrade pip

# --- 2. PyTorch first ---
pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu126

python -c "
import torch
assert torch.cuda.is_available(), 'ERROR: Torch cannot see GPU'
print('Torch OK | version:', torch.__version__, '| CUDA:', torch.version.cuda)
"

# # --- 3. JAX ---
pip install "jax[cuda12]==0.5.3"

python -c "
import jax
devs = jax.devices()
print('JAX OK | version:', jax.__version__, '| devices:', devs)
assert any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in devs), \
    'ERROR: JAX cannot see GPU'
"


# --- 7. av via conda-forge (fixes the ffmpeg build error from uv) ---
conda install -c conda-forge "av=14.4.0" -y --freeze-installed

# --- 4. Pre-pin the packages that cause resolver explosion ---
# These are the exact versions that openpi + lerobot are compatible with.
# Installing them first collapses the search space before the editable install.

# DO NOT NEED LEROBOT FOR THE SERVER SIDE
pip install \
    "numpy==1.26.4" \
    "datasets==3.6.0" \
    "diffusers==0.27.2" \
    "huggingface-hub==0.35.0" \
    "accelerate==1.10.0" \
    "dill==0.3.8" \
    "multiprocess==0.70.16" \
    "gymnasium==0.29.1" \
    "mujoco==3.3.2" \
    "dm-control==1.0.26" \
    "opencv-python-headless==4.11.0.86" \
    "grpcio==1.67.0" \
    "grpcio-status==1.67.0" \
    "protobuf==5.29.3" \
    "packaging==24.2" \
    "torchcodec==0.5" \
    "deepdiff==8.6.0" \
    "draccus==0.10.0" \
    "cmake==4.0.3"

# --- 5. Now install openpi editable — resolver has almost nothing to decide ---
pip install -e ./openpi --no-deps
pip install -e ./openpi/packages/openpi-client --no-deps

# --- 6. Install openpi's own direct deps (from its pyproject.toml) explicitly ---
# Using --no-deps above means we must install these ourselves.
pip install \
    flax==0.10.2 \
    optax==0.2.4 \
    orbax-checkpoint==0.11.13 \
    chex==0.1.89 \
    equinox==0.12.2 \
    augmax==0.4.1 \
    transformers==4.53.2 \
    safetensors==0.5.3 \
    tokenizers==0.21.1 \
    sentencepiece==0.2.0 \
    pillow==11.2.1 \
    einops==0.8.1 \
    wandb==0.19.11 \
    tyro==0.9.22 \
    flask==3.1.1 \
    websockets==15.0.1 \
    imageio==2.37.0 \
    imageio-ffmpeg==0.6.0 \
    zarr==3.0.8 \
    h5py==3.13.0 \
    rich==14.0.0 \
    tqdm==4.67.1 \
    "tqdm-loggable>=0.2" \
    pydantic==2.11.5 \
    jaxtyping==0.2.36 \
    beartype==0.19.0 \
    ml_collections==1.0.0 \
    "numpydantic>=1.6.6" \
    "polars>=1.30.0" \
    "treescope>=0.1.7" \
    "fsspec[gcs]>=2024.6.0" \
    "typing-extensions>=4.12.2" \
    dm-tree \
    flatbuffers \
    msgpack

pip install pytest
# --- 8. Patch transformers (required for pi0.5) ---
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cp -r ./openpi/src/openpi/models_pytorch/transformers_replace/* \
    "${SITE_PACKAGES}/transformers/"
echo "Patch applied to: ${SITE_PACKAGES}/transformers/"

pip install poetry-core
pip install --no-build-isolation --no-deps \
    "git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"

pip install \
    "einops>=0.8.0" \
    "jsonlines>=4.0.0" \
    "pyserial>=3.5" \
    "wandb>=0.19.1" \
    "termcolor>=2.4.0" \
    "pynput>=1.7.7"
# --- 9. Final verification ---

python -c "
import torch; print('torch:', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))
import jax; print('jax:', jax.__version__, '| devices:', jax.devices())
import flax; print('flax:', flax.__version__)
import openpi; print('openpi: OK')
import av; print('av:', av.__version__)
"
python -c "import lerobot; print('lerobot OK:', lerobot.__version__)"
echo "=== Setup complete ==="
echo "To serve: export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9"
echo "          python openpi/scripts/serve_policy.py --env LIBERO"