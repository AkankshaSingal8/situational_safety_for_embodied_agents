#!/usr/bin/env bash
# Adds lightweight policy-client deps (websockets, openpi-client, imageio) to
# the container-built `oopsieverse_b1k` conda env, so the B1K eval client can
# talk to the pi0.5 websocket server and save rollout videos without pulling
# any ML stack into the env that hosts Isaac Sim/OmniGibson. Mirrors
# `setup_robocasa_client_deps.sh`'s exact approach (openpi-client installed
# --no-deps to avoid its numpy<2.0.0 pin fighting this env's real numpy;
# confirmed via `import openpi_client, websockets, imageio` failing with
# ModuleNotFoundError before this script runs -- see
# oopsieverse_eval/eval/run_b1k_pi05_eval.py's need for these at import time).
#
# MUST run inside the b1k_container.sif Apptainer container (this env was
# built with the container's own Miniconda, not host conda) -- run via
# `apptainer exec` (no --nv/GPU needed, this is pure pip installs, so no
# SLURM GPU allocation required either):
#
#   apptainer exec \
#       --bind /ocean/projects/cis250185p/asingal:/ocean/projects/cis250185p/asingal \
#       oopsieverse_eval/setup/b1k_container.sif \
#       bash oopsieverse_eval/setup/setup_b1k_client_deps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_CLIENT_DIR="$SCRIPT_DIR/../external/openpi/packages/openpi-client"
ENV_NAME="oopsieverse_b1k"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

pip install dm-tree msgpack pillow websockets imageio

if [ ! -d "$OPENPI_CLIENT_DIR" ]; then
    echo "ERROR: $OPENPI_CLIENT_DIR not found -- run setup_pi05_server_env.sh first (clones openpi)."
    exit 1
fi
pip install -e "$OPENPI_CLIENT_DIR" --no-deps

python -c "
from openpi_client import websocket_client_policy, image_tools
import imageio
print('openpi_client + imageio: OK')
"
