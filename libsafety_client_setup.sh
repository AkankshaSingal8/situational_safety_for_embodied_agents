#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/libsafety_client"
LIBSAFETY_DIR="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/LIBERO-Safety"
OPENPI_CLIENT_DIR="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/vlsa-aegis/openpi/packages/openpi-client"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -p "${ENV_PREFIX}" python=3.8.13 -y
conda activate "${ENV_PREFIX}"

pip install --upgrade pip

pip install \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

conda install -c conda-forge "av>=9.0.0" -y --freeze-installed

# Editable install of the LIBERO-Safety repo itself (per its own README:
# pip install -e ., then requirements.txt, extra_requirements.txt, robosuite)
cd "${LIBSAFETY_DIR}"
pip install -e .
pip install -r requirements.txt
pip install -r extra_requirements.txt
pip install -e third_party/robosuite-1.4

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

python -c "
import robosuite; print('robosuite: OK')
from libero.libero.envs import OffScreenRenderEnv; print('libero envs: OK')
import openpi_client; print('openpi_client: OK')
"
