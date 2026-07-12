#!/usr/bin/env bash
# Adds lightweight policy-client deps (requests, json-numpy, openpi-client)
# to the already-built `oopsieverse_robocasa` env, so the sim-host can talk
# to the OpenVLA REST server and the pi0.5 websocket server without pulling
# either policy's heavier ML stack into this env. Run this AFTER
# setup_pi05_server_env.sh (which clones openpi, providing openpi-client).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_CLIENT_DIR="$SCRIPT_DIR/../external/openpi/packages/openpi-client"
ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/oopsieverse_robocasa"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

pip install requests json-numpy

# openpi-client's own pyproject.toml pins numpy<2.0.0, which would conflict
# with oopsieverse_robocasa's robosuite/robocasa-driven numpy 2.x -- install
# its actual runtime deps (dm-tree, msgpack, pillow, websockets) unpinned,
# then the package itself with --no-deps, rather than letting pip's resolver
# try to satisfy openpi-client's numpy<2 pin against this env's real numpy.
pip install dm-tree msgpack pillow websockets
pip install -e "$OPENPI_CLIENT_DIR" --no-deps

python -c "
import requests, json_numpy; print('requests/json_numpy: OK')
from openpi_client import websocket_client_policy, image_tools
print('openpi_client: OK')
"
