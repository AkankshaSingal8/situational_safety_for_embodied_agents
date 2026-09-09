#!/usr/bin/env bash
# Run this script on a GPU node (interactive or batch) to create fastwam_env.
# Usage: bash fast-wam/setup_fastwam_env.sh
set -euo pipefail

###############################################################################
# PATHS
###############################################################################
REPO_ROOT="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents"
ENV_NAME="fastwam_env"
ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/${ENV_NAME}"
FASTWAM_REPO="${REPO_ROOT}/fast-wam/FastWAM"
SAFELIBERO_REPO="${REPO_ROOT}/SafeLIBERO/safelibero"

###############################################################################
# CONDA INIT
###############################################################################
echo "============================================================"
echo " fastwam_env setup — $(date)"
echo "============================================================"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo ""
echo "[1/7] Creating conda env '${ENV_NAME}' at ${ENV_PREFIX} (Python 3.10)..."
conda create -p "${ENV_PREFIX}" python=3.10 -y

echo ""
echo "[2/7] Activating env..."
conda activate "${ENV_PREFIX}"

python -m pip install --upgrade pip setuptools wheel

###############################################################################
# PYTORCH — cu128 index
###############################################################################
echo ""
echo "[3/7] Installing PyTorch 2.7.1+cu128 stack..."
pip install \
  torch==2.7.1+cu128 \
  torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

###############################################################################
# FASTWAM PACKAGE (editable) + ALL pyproject.toml DEPS
# deepspeed: skip CUDA-op compilation on HPC (DS_BUILD_OPS=0)
###############################################################################
echo ""
echo "[4/7] Installing FastWAM package (editable) and its dependencies..."

if [ ! -d "${FASTWAM_REPO}" ]; then
  echo "ERROR: FastWAM repo not found at ${FASTWAM_REPO}"
  exit 1
fi

# torchcodec may need a separate index; install it separately first so a
# failure doesn't block the rest.  It is used for video decoding and is
# optional for eval-only runs.
set +e
pip install torchcodec==0.5 --index-url https://download.pytorch.org/whl/cu128 2>/dev/null \
  || pip install torchcodec==0.5 2>/dev/null \
  || echo "WARNING: torchcodec could not be installed — video decoding may be unavailable."
set -e

# Install FastWAM dependencies with exact versions from pyproject.toml,
# excluding torch/torchvision/torchcodec (already handled above).
# DS_BUILD_OPS=0 tells DeepSpeed not to compile CUDA extensions at install time.
DS_BUILD_OPS=0 pip install \
  accelerate==1.12.0 \
  av==16.0.1 \
  boto3==1.35.99 \
  datasets==3.6.0 \
  deepspeed==0.18.5 \
  einops==0.8.1 \
  gitpython==3.1.45 \
  huggingface-hub==0.29.2 \
  hydra-core==1.3.2 \
  imageio==2.37.0 \
  imageio-ffmpeg==0.6.0 \
  jsonlines==4.0.0 \
  modelscope==1.34.0 \
  numpy==1.26.4 \
  omegaconf==2.3.0 \
  packaging==25.0 \
  pandas==2.2.3 \
  pillow==12.0.0 \
  pyarrow==23.0.0 \
  regex==2025.11.3 \
  rich==14.2.0 \
  safetensors==0.5.3 \
  termcolor==2.5.0 \
  tqdm==4.66.5 \
  transformers==4.49.0 \
  typing-extensions==4.15.0 \
  wandb==0.23.1

# Editable install of FastWAM itself (pulls remaining deps from pyproject.toml)
DS_BUILD_OPS=0 pip install -e "${FASTWAM_REPO}"

###############################################################################
# ADDITIONAL EVAL DEPS (not in pyproject.toml)
###############################################################################
echo ""
echo "[5/7] Installing additional eval dependencies..."
pip install \
  opencv-python-headless \
  scipy \
  tqdm  # already pinned above but harmless

###############################################################################
# LIBERO / SAFELIBERO (needed for single-process eval)
###############################################################################
echo ""
echo "[6/7] Attempting LIBERO install for single-process eval..."

# robosuite and bddl are direct LIBERO dependencies
set +e
pip install \
  robosuite==1.4.1 \
  bddl==1.0.1 \
  easydict==1.9 \
  future==0.18.2 \
  cloudpickle==2.1.0 \
  gym==0.25.2
LIBERO_DEPS_STATUS=$?
set -e

if [ $LIBERO_DEPS_STATUS -ne 0 ]; then
  echo "WARNING: Some LIBERO deps failed to install. LIBERO import may not work."
fi

# Install the SafeLIBERO package itself
if [ -d "${SAFELIBERO_REPO}" ]; then
  set +e
  pip install -e "${SAFELIBERO_REPO}"
  SAFELIBERO_STATUS=$?
  set -e
  if [ $SAFELIBERO_STATUS -ne 0 ]; then
    echo "WARNING: SafeLIBERO editable install failed — continuing without it."
    echo "         You may need to add ${SAFELIBERO_REPO} to PYTHONPATH manually."
  else
    echo "SafeLIBERO installed OK."
  fi
else
  echo "WARNING: SafeLIBERO repo not found at ${SAFELIBERO_REPO} — skipping."
fi

###############################################################################
# VERIFICATION
###############################################################################
echo ""
echo "[7/7] Running verification checks..."

python - <<'PY'
import sys
print("Python:", sys.version)

import torch
print("PyTorch:", torch.__version__)
cuda_ok = torch.cuda.is_available()
print("CUDA available:", cuda_ok)
if cuda_ok:
    print("GPU device:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA not available — make sure you are on a GPU node.")

import torchvision
print("torchvision:", torchvision.__version__)

# FastWAM core import
try:
    import fastwam
    print("fastwam: import OK  (version:", getattr(fastwam, "__version__", "n/a"), ")")
except Exception as e:
    print("ERROR: fastwam import FAILED:", e)
    sys.exit(1)

# Key dep spot-checks
for mod in [
    "accelerate", "transformers", "hydra", "omegaconf", "einops",
    "deepspeed", "wandb", "huggingface_hub", "numpy", "PIL",
    "cv2", "scipy", "imageio", "tqdm",
]:
    try:
        __import__(mod)
        print(f"  {mod}: OK")
    except ImportError as e:
        print(f"  {mod}: MISSING ({e})")

# LIBERO (optional for this env — graceful fallback at runtime)
try:
    import libero
    print("libero: import OK")
except Exception as e:
    print("libero: not importable —", e)
    print("  (Set PYTHONPATH or re-run after fixing LIBERO install if needed.)")

PY

echo ""
echo "============================================================"
echo " Setup complete!"
echo " Activate with:"
echo "   conda activate ${ENV_PREFIX}"
echo ""
echo " If LIBERO import failed above, add to your PYTHONPATH:"
echo "   export PYTHONPATH=${SAFELIBERO_REPO}:\$PYTHONPATH"
echo ""
echo " NOTE: DeepSpeed was installed with DS_BUILD_OPS=0 (no CUDA"
echo " op compilation). If you need compiled CUDA ops at runtime,"
echo " re-run:  DS_BUILD_OPS=1 pip install deepspeed==0.18.5"
echo "============================================================"
