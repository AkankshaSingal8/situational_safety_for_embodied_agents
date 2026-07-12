#!/usr/bin/env bash
# Creates a fresh, isolated `oopsieverse_pi05_srv` conda env for serving the
# off-the-shelf `pi05_libero` checkpoint via openpi's own (unmodified)
# scripts/serve_policy.py.
#
# This is a brand-new env against a brand-new openpi checkout
# (oopsieverse_eval/external/openpi), independent of the sibling
# LIBERO-Safety benchmark's `libsafety_openpi` env and its
# vlsa-aegis/openpi checkout -- neither is touched or imported from. Package
# pins below are taken from that env's own known-working recipe
# (libsafety_openpi_setup.sh) purely as a reference for versions that are
# already confirmed to resolve cleanly on this cluster; nothing here reads
# from or writes into that env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_REPO="$SCRIPT_DIR/../external/openpi"
ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/oopsieverse_pi05_srv"

if [ ! -d "$OPENPI_REPO" ]; then
  echo "Cloning openpi into $OPENPI_REPO ..."
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$OPENPI_REPO"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -p "${ENV_PREFIX}" python=3.11 -y
conda activate "${ENV_PREFIX}"

# `av` has no prebuilt wheel for this platform/python combo and builds from
# source, which needs ffmpeg's dev libraries visible via pkg-config.
conda install -y -c conda-forge ffmpeg=7.1.1
export PKG_CONFIG_PATH="${ENV_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install "jax[cuda12]"==0.5.3

# Pre-install openpi's full pinned dependency closure, then install
# openpi/openpi-client/lerobot with --no-deps, to sidestep pip resolver
# backtracking failures between openpi's loose numpy<2 pins and
# unpinned transitive deps (augmax/rerun-sdk) that otherwise want numpy>=2.
pip install \
  flax==0.10.2 optax==0.2.4 orbax-checkpoint==0.11.13 chex==0.1.89 \
  equinox==0.12.2 augmax==0.4.1 \
  diffusers==0.33.1 transformers==4.53.2 safetensors==0.5.3 \
  tokenizers==0.21.1 sentencepiece==0.2.0 \
  opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86 pillow==11.2.1 \
  numpy==1.26.4 scipy==1.15.3 einops==0.8.1 \
  wandb==0.19.11 \
  draccus==0.10.0 tyro==0.9.22 omegaconf==2.3.0 \
  flask==3.1.1 websockets==15.0.1 \
  huggingface-hub==0.32.3 datasets==3.6.0 \
  av==14.4.0 imageio==2.37.0 imageio-ffmpeg==0.6.0 \
  dm-control==1.0.14 mujoco==2.3.7 \
  zarr==3.0.8 h5py==3.13.0 \
  rich==14.0.0 tqdm==4.67.1 \
  pydantic==2.11.5 jaxtyping==0.2.36 beartype==0.19.0 \
  dm-tree==0.1.10 flatbuffers==25.12.19 ml_collections==1.0.0 \
  numpydantic==1.8.0 tqdm-loggable==0.4.1 typing_extensions==4.15.0 \
  filelock==3.25.2 treescope==0.1.10 polars==1.39.3 msgpack==1.1.2 \
  fsspec==2025.3.0 gcsfs==2025.3.0

pip install --no-deps "lerobot@git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
pip install -e "${OPENPI_REPO}" --no-deps
pip install -e "${OPENPI_REPO}/packages/openpi-client" --no-deps

# `lerobot`'s own dataset utils import `jsonlines` at module level, but it's
# not in the pinned closure above (only surfaced when actually running
# serve_policy.py, which imports through openpi -> lerobot's data_loader).
pip install jsonlines

# Route checkpoint/config downloads to $OCEAN, NOT $HOME -- home quota on
# this cluster is nearly full (~1.6GB free as of this build).
export OPENPI_DATA_HOME="/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/openpi"
mkdir -p "$OPENPI_DATA_HOME"

python -c "import jax; print('jax devices:', jax.devices())"
python -c "import openpi; print('openpi: OK')"
python -c "import openpi_client; print('openpi_client: OK')"
echo "oopsieverse_pi05_srv env ready. Remember to set OPENPI_DATA_HOME=$OPENPI_DATA_HOME when serving."
