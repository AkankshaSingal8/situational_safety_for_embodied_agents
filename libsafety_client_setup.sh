#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/libsafety_client"
LIBSAFETY_DIR="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/LIBERO-Safety"
OPENPI_CLIENT_DIR="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/vlsa-aegis/openpi/packages/openpi-client"

source "$(conda info --base)/etc/profile.d/conda.sh"

# python=3.8.13 (matching the older SafeLIBERO/libero_env convention) does
# NOT work here: LIBERO-Safety's own libero/libero/benchmark/__init__.py
# uses PEP 604 union syntax (`Task | None`) in a function signature that
# gets evaluated at class-definition time, which raises TypeError on
# Python < 3.10. This only surfaces when something actually imports
# libero.libero.benchmark (e.g. benchmark.get_benchmark_dict(), used by
# every eval driver) -- plain `from libero.libero.envs import
# OffScreenRenderEnv` does NOT trigger it, which is why this wasn't caught
# by this script's own (narrower) verification the first time around.
conda create -p "${ENV_PREFIX}" python=3.10 -y
conda activate "${ENV_PREFIX}"

pip install --upgrade pip

pip install \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117 || \
pip install \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118

conda install -c conda-forge "av>=9.0.0" -y --freeze-installed

# LIBERO-Safety's requirements.txt pins robomimic==0.2.0, which itself pins
# opencv-python==4.6.0.66 -- a wheel built against numpy<2's C ABI. Neither
# requirements.txt nor extra_requirements.txt pins numpy itself, so pip's
# resolver picks the latest numpy (2.x) satisfying everyone else's loose
# constraints, and `import cv2` then fails with "numpy.core.multiarray
# failed to import" (an ABI break, not a missing-package error). Pin numpy
# explicitly first so the opencv-python==4.6.0.66 resolved below is
# actually numpy-2-free.
pip install "numpy==1.26.4"

# Editable install of the LIBERO-Safety repo itself (per its own README:
# pip install -e ., then requirements.txt, extra_requirements.txt, robosuite)
cd "${LIBSAFETY_DIR}"
pip install -e .
pip install -r requirements.txt
pip install -r extra_requirements.txt
pip install -e third_party/robosuite-1.4
pip install "numpy==1.26.4"

pip install -e "${OPENPI_CLIENT_DIR}" --no-deps
pip install "cachetools>=4.2.2" "httpx>=0.23.0" "websockets==13.1"

export MUJOCO_GL=egl

# libero's __init__.py interactively prompts ("custom path for the dataset
# folder? (Y/N)") the first time it imports and finds no config file at
# LIBERO_CONFIG_PATH/config.yaml. Under sbatch there is no stdin, so this
# would hang (or EOFError) any job that imports libero for the first time —
# including the later eval driver (Task 8), not just this script. Avoid the
# prompt entirely (durably, for all future jobs using this env) by
# pre-creating the config directory with a valid config.yaml pointing at
# LIBERO-Safety's own paths before the first `import libero`. Note:
# LIBERO_CONFIG_PATH is a *directory* (containing config.yaml), not a path
# to a yaml file itself.
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
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"

export LD_PRELOAD="${ENV_PREFIX}/lib/libstdc++.so.6"
python -c "
import robosuite; print('robosuite: OK')
from libero.libero.envs import OffScreenRenderEnv; print('libero envs: OK')
from libero.libero import benchmark; d = benchmark.get_benchmark_dict(); print('libero benchmark: OK', list(d.keys()))
import openpi_client; print('openpi_client: OK')
"
