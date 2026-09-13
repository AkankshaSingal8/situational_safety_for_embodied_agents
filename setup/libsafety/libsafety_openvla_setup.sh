#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/libsafety_openvla"
LIBSAFETY_DIR="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/LIBERO-Safety"
OPENVLA_OFT_REPO="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/openvla-oft"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -p "${ENV_PREFIX}" python=3.10 -y
conda activate "${ENV_PREFIX}"

pip install --upgrade pip setuptools wheel

pip install \
  torch==2.2.0 \
  torchvision==0.17.0 \
  torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu118

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

cd "${LIBSAFETY_DIR}"
###############################################################################
# Upstream LIBERO-Safety packaging bug workaround
#
# setup.py does `packages=[p for p in find_packages() if p.startswith("libero")]`,
# but unlike the sibling SafeLIBERO repo (which ships an empty
# `libero/__init__.py` at the top level), this checkout has no top-level
# `libero/__init__.py`. Without it, `find_packages()` cannot discover
# `libero.libero`, `libero.configs`, `libero.lifelong` as installable
# packages at all, so `pip install -e .` "succeeds" but produces a hollow
# editable install (empty MAPPING in the generated finder) and
# `import libero` fails with ModuleNotFoundError. Create the missing empty
# `__init__.py`, matching the sibling repo's structure, before installing.
###############################################################################
touch "${LIBSAFETY_DIR}/libero/__init__.py"
pip install -e .
pip install -r requirements.txt
pip install -r extra_requirements.txt
pip install -e third_party/robosuite-1.4

cd "${OPENVLA_OFT_REPO}"
pip install -e .

pip install tensorflow-addons==0.23.0
pip install "typeguard>=4,<5"
pip install tyro==0.9.2

###############################################################################
# protobuf version fix
#
# openvla-oft's own (unpinned) "protobuf" dependency lets pip's resolver
# land on protobuf==4.25.9 here, but tensorflow_datasets==4.9.3 pulls in
# tensorflow-metadata==1.21.0, whose generated *_pb2.py files import
# `google.protobuf.runtime_version` -- a module that only exists in
# protobuf>=5.26. Without it, `import dlimp` (transitively imported by
# openvla-oft's prismatic.vla.datasets, which experiments.robot.robot_utils
# imports) fails with:
#   ImportError: cannot import name 'runtime_version' from 'google.protobuf'
# The known-working `openvla_libero_merged` env (same torch/tensorflow
# stack) has protobuf==5.29.6 installed for exactly this reason. Pin to the
# same version here.
#
# That alone isn't sufficient though: tensorflow_datasets==4.9.3's
# unpinned tensorflow-metadata dependency resolves to the latest available
# release (1.21.0 as of this writing), whose generated *_pb2.py files were
# compiled against a newer protoc (gencode 6.31.1) than protobuf==5.29.6's
# runtime supports, producing:
#   google.protobuf.runtime_version.VersionError: Detected incompatible
#   Protobuf Gencode/Runtime versions ... gencode 6.31.1 runtime 5.29.6.
# The known-working `openvla_libero_merged` env has tensorflow-metadata
# pinned at 1.17.3 (gencode compatible with protobuf 5.29.6), so pin the
# same version here instead of letting the resolver pick the newer one.
###############################################################################
pip install protobuf==5.29.6 tensorflow-metadata==1.17.3

###############################################################################
# LIBERO_CONFIG_PATH fix
#
# LIBERO-Safety's `libero/libero/__init__.py` treats LIBERO_CONFIG_PATH as a
# *directory* containing a `config.yaml` (default: ~/.libero), NOT a path to
# the yaml file itself. If that config.yaml doesn't already exist, the
# `import libero` machinery calls `input()` to interactively ask about a
# custom dataset path -- which hangs forever with no stdin (as in an sbatch
# job). Pre-create the config directory and its config.yaml, pointing at
# this checkout's own bddl_files/init_files/assets/datasets, so `import
# libero` finds it already populated and never prompts.
###############################################################################
LIBERO_CONFIG_DIR="${LIBSAFETY_DIR}/.libero_config"
mkdir -p "${LIBERO_CONFIG_DIR}"
mkdir -p "${LIBSAFETY_DIR}/libero/datasets"
cat > "${LIBERO_CONFIG_DIR}/config.yaml" << YAML
assets: ${LIBSAFETY_DIR}/libero/libero/assets
bddl_files: ${LIBSAFETY_DIR}/libero/libero/bddl_files
benchmark_root: ${LIBSAFETY_DIR}/libero/libero
datasets: ${LIBSAFETY_DIR}/libero/datasets
init_states: ${LIBSAFETY_DIR}/libero/libero/init_files
YAML

export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"
python -c "
import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
import robosuite; print('robosuite: OK')
import libero; print('libero: OK')
from experiments.robot.robot_utils import get_model; print('openvla-oft robot_utils: OK')
"
