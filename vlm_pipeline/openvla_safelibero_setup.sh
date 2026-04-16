#Run after getting GPU
#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# CONFIGURE THESE PATHS
###############################################################################
ENV_NAME="openvla_libero_merged"
ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/${ENV_NAME}"

# Existing local repos
OPENVLA_OFT_REPO="/ocean/projects/cis250185p/asingal/openvla-oft"
SAFELIBERO_REPO="/ocean/projects/cis250185p/asingal/SafeLIBERO/safelibero"

###############################################################################
# CONDA INIT
###############################################################################
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Creating env at ${ENV_PREFIX}"
conda create -p "${ENV_PREFIX}" python=3.10 -y
conda activate "${ENV_PREFIX}"

python -m pip install --upgrade pip setuptools wheel

###############################################################################
# PYTORCH STACK (matches your current working openvla-oft side)
###############################################################################
pip install \
  torch==2.2.0 \
  torchvision==0.17.0 \
  torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu118

###############################################################################
# OPENVLA / GENERAL DEPS
###############################################################################
pip install \
  accelerate>=0.25.0 \
  draccus==0.8.0 \
  einops==0.4.1 \
  huggingface_hub \
  json-numpy \
  jsonlines \
  matplotlib \
  peft==0.11.1 \
  protobuf \
  rich \
  sentencepiece==0.1.99 \
  timm==0.9.10 \
  tokenizers==0.19.1 \
  wandb \
  tensorflow==2.15.0 \
  tensorflow_datasets==4.9.3 \
  tensorflow_graphics==2021.12.3 \
  diffusers==0.30.3 \
  imageio \
  uvicorn \
  fastapi \
  packaging \
  ninja

###############################################################################
# SAFELIBERO / LIBERO DEPS
###############################################################################
pip install \
  hydra-core==1.2.0 \
  numpy==1.24.4 \
  wandb==0.25.1 \
  easydict==1.9 \
  opencv-python==4.6.0.66 \
  robomimic==0.2.0 \
  thop==0.1.1-2209072238 \
  robosuite==1.4.1 \
  bddl==1.0.1 \
  future==0.18.2 \
  cloudpickle==2.1.0 \
  gym==0.25.2 \
  tqdm

###############################################################################
# IMPORTANT: tyro/typeguard conflict workaround
#
# Your logs showed:
# - tensorflow-addons pulls typeguard<3
# - tyro==0.9.2 needs typeguard>=4
#
# So install tensorflow-addons first, then force typeguard>=4, then tyro.
###############################################################################
pip install tensorflow-addons==0.23.0
pip install "typeguard>=4,<5"
pip install tyro==0.9.2

###############################################################################
# INSTALL EXISTING LOCAL REPOS
###############################################################################
if [ ! -d "${OPENVLA_OFT_REPO}" ]; then
  echo "ERROR: OPENVLA_OFT_REPO not found: ${OPENVLA_OFT_REPO}"
  exit 1
fi

# if [ ! -d "${AEGIS_REPO}" ]; then
#   echo "ERROR: AEGIS_REPO not found: ${AEGIS_REPO}"
#   exit 1
# fi

cd "${OPENVLA_OFT_REPO}"
pip install -e .

# # Optional; only if vlsa-aegis itself is a Python package
# cd "${SAFELIBERO_REPO}"
# pip install -e . || true



###############################################################################
# OPTIONAL FLASH-ATTN
###############################################################################
set +e
pip cache remove flash_attn >/dev/null 2>&1 || true
pip install "flash-attn==2.5.5" --no-build-isolation
FA_STATUS=$?
set -e

if [ $FA_STATUS -ne 0 ]; then
  echo "FlashAttention install failed; continuing without it."
fi

###############################################################################
# SANITY CHECKS
###############################################################################
python - <<'PY'
import sys
print("Python:", sys.version)

import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

mods = [
    "draccus",
    "tyro",
    "tensorflow",
    "tensorflow_datasets",
    "robosuite",
    "wandb",
]
for m in mods:
    __import__(m)
    print("Imported:", m)

try:
    import libero
    print("Imported: libero")
except Exception as e:
    print("Could not import libero:", e)

try:
    from experiments.robot.robot_utils import get_model
    print("Imported openvla-oft robot_utils OK")
except Exception as e:
    print("Could not import openvla-oft robot_utils:", e)
PY

echo
echo "Done."
echo "Activate with:"
echo "  conda activate ${ENV_PREFIX}"

#Error fixes
pip install "typeguard>=4,<5" "tyro==0.9.2"
pip install "setuptools<70"