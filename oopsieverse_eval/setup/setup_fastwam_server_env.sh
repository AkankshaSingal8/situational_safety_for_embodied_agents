#!/usr/bin/env bash
# Creates a fresh, isolated `oopsieverse_fastwam_srv` conda env for serving
# the off-the-shelf Fast-WAM checkpoint (yuanty/fastwam, libero_uncond_2cam224)
# over a small REST API (see ../servers/fastwam_server.py).
#
# Unlike the existing `fastwam_env` in this repo (which co-installs
# robosuite/bddl/SafeLIBERO into the SAME env for single-process eval), this
# env installs ONLY the model stack -- no sim deps at all. Reason: OopsieVerse's
# RoboCasa fork needs robosuite==1.5.2, while fastwam_env pins robosuite==1.4.1
# (SafeLIBERO's requirement) -- these can't coexist in one env, so serving
# Fast-WAM over localhost to the already-built `oopsieverse_robocasa` env is
# the only way to use both without a version conflict. Package pins below are
# taken from fast-wam/setup_fastwam_env.sh purely as a reference recipe --
# nothing here touches that env or its checkpoint download location.
set -euo pipefail

# `fast-wam/FastWAM` is gitignored in the main repo (see fast-wam/.gitignore)
# -- it's a large local checkout, not tracked git content -- so it does NOT
# exist inside this (or any) worktree, only in the main repo's working
# directory. Reference that fixed absolute path directly (read-only), the
# same way the existing Cosmos SLURM jobs reference cosmos-policy's own
# untracked build artifacts (.venv_rhel8, cosmos_policy.sif) by absolute
# main-repo path rather than deriving from the worktree.
MAIN_REPO="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents"
FASTWAM_REPO="${MAIN_REPO}/fast-wam/FastWAM"
ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/oopsieverse_fastwam_srv"

if [ ! -d "${FASTWAM_REPO}" ]; then
  echo "ERROR: FastWAM repo not found at ${FASTWAM_REPO} -- this reuses the" \
       "existing vendored fast-wam/FastWAM source (read-only), it is not cloned by this script."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -p "${ENV_PREFIX}" python=3.10 -y
conda activate "${ENV_PREFIX}"

python -m pip install --upgrade pip setuptools wheel

pip install \
  torch==2.7.1+cu128 \
  torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

set +e
pip install torchcodec==0.5 --index-url https://download.pytorch.org/whl/cu128 2>/dev/null \
  || pip install torchcodec==0.5 2>/dev/null \
  || echo "WARNING: torchcodec could not be installed -- video decoding may be unavailable (not needed for eval-only serving)."
set -e

# DS_BUILD_OPS=0: skip DeepSpeed CUDA-op compilation, matching fastwam_env's
# own approach on this cluster.
DS_BUILD_OPS=0 pip install \
  accelerate==1.12.0 \
  av==16.0.1 \
  datasets==3.6.0 \
  deepspeed==0.18.5 \
  einops==0.8.1 \
  huggingface-hub==0.29.2 \
  hydra-core==1.3.2 \
  imageio==2.37.0 \
  imageio-ffmpeg==0.6.0 \
  jsonlines==4.0.0 \
  numpy==1.26.4 \
  omegaconf==2.3.0 \
  packaging==25.0 \
  pillow==12.0.0 \
  regex==2025.11.3 \
  rich==14.2.0 \
  safetensors==0.5.3 \
  termcolor==2.5.0 \
  tqdm==4.66.5 \
  transformers==4.49.0 \
  typing-extensions==4.15.0

DS_BUILD_OPS=0 pip install -e "${FASTWAM_REPO}"

pip install opencv-python-headless scipy fastapi "uvicorn[standard]" json-numpy

# Point at the ALREADY-DOWNLOADED Wan2.2-TI2V-5B backbone (~22GB) and
# Fast-WAM checkpoint (~12GB) read-only -- these are data artifacts (like an
# HF checkpoint), not env/code, so re-downloading ~34GB here would be pure
# waste with no isolation benefit. Nothing in this script writes into either
# path. If either is ever missing, re-run fast-wam/download_checkpoint.sh
# (unmodified) to (re)populate them.
export HF_HOME="/ocean/projects/cis250185p/asingal/.hf_cache"
export DIFFSYNTH_MODEL_BASE_PATH="${MAIN_REPO}/fast-wam/checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE="huggingface"

python -c "
import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
import fastwam; print('fastwam: OK')
import hydra, omegaconf; print('hydra/omegaconf: OK')
import fastapi, uvicorn, json_numpy; print('server deps: OK')
"
echo "oopsieverse_fastwam_srv env ready."
echo "Remember to set these when serving:"
echo "  HF_HOME=$HF_HOME"
echo "  DIFFSYNTH_MODEL_BASE_PATH=$DIFFSYNTH_MODEL_BASE_PATH"
echo "  DIFFSYNTH_DOWNLOAD_SOURCE=huggingface"
