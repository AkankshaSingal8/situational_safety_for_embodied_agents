#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/libsafety_openpi"
OPENPI_REPO="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/vlsa-aegis/openpi"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -p "${ENV_PREFIX}" python=3.11 -y
conda activate "${ENV_PREFIX}"

# `av==14.4.0` (below) has no prebuilt wheel for this platform/python combo
# and falls back to building from source, which requires ffmpeg's dev
# libraries (libavformat, libavcodec, ...) discoverable via pkg-config.
# Install ffmpeg from conda-forge first, matching the known-working `aegis`
# env, so the `av` build succeeds.
conda install -y -c conda-forge ffmpeg=7.1.1

# The system pkg-config (/usr/bin/pkg-config) doesn't know about the conda
# env's .pc files unless PKG_CONFIG_PATH points at them explicitly, so the
# `av` build below can't find libavformat.pc etc. even though ffmpeg is
# installed. Point it at the env's pkgconfig dir.
export PKG_CONFIG_PATH="${ENV_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install "jax[cuda12]"==0.5.3

# Install openpi's full (pinned) dependency closure BEFORE the editable
# installs below, then install openpi/openpi-client/lerobot with --no-deps.
#
# Why: openpi's pyproject.toml pins numpy<2.0.0 but leaves several deps
# (e.g. augmax>=0.3.4) unpinned. When pip resolves `pip install -e openpi`
# itself, its backtracking resolver can select augmax==0.4.0 (which requires
# numpy>=2.0.0), producing a ResolutionImpossible error even though a valid
# combination (augmax==0.4.1 + numpy==1.26.4, as installed below) exists.
# Pre-installing pinned versions and then using --no-deps for the editable
# installs sidesteps this resolver backtracking entirely. Versions below
# were taken from the known-working `aegis` conda env (pip freeze), which
# uses the same openpi checkout.
#
# `lerobot` (installed from git below) is ALSO installed with --no-deps, for
# the same reason: lerobot's own dependency spec pulls in an unpinned
# `rerun-sdk`, and every currently-available rerun-sdk release requires
# numpy>=2 — a hard conflict with numpy==1.26.4 that a single combined `pip
# install` cannot resolve (confirmed via the resolver's "conflict is caused
# by" trace: augmax/chex/diffusers/flax/numpy/opencv/optax/orbax-checkpoint/
# scipy/transformers all pin numpy<2 while every rerun-sdk release, pulled
# in transitively by lerobot, requires numpy>=2). lerobot's actual runtime
# deps are already covered by the explicit pins in this same block (taken
# from the aegis env's pip freeze, which has this identical lerobot commit
# installed and working), so --no-deps here just skips lerobot's own
# (broken) dependency resolution rather than skipping any real requirement.
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

python -c "import jax; print('jax devices:', jax.devices())"
python -c "import openpi; print('openpi: OK')"
