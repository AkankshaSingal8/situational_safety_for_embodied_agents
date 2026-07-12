#!/usr/bin/env bash
# Thin wrapper around openpi's own scripts/serve_policy.py -- no custom
# server code needed for pi0.5, openpi already ships exactly this tool.
#
# Usage: bash launch_pi05_server.sh [port] [checkpoint_dir]
#
# Must run inside the `oopsieverse_pi05_srv` conda env (see
# ../setup/setup_pi05_server_env.sh). `--policy.dir` defaults to openpi's own
# public GCS release of pi05_libero (per openpi's own
# scripts/serve_policy.py DEFAULT_CHECKPOINT table) -- openpi downloads and
# caches it itself (via gcsfs, into OPENPI_DATA_HOME) on first run, no manual
# download step needed. Pass a local dir as $2 to serve an already-downloaded
# copy instead.
set -euo pipefail

PORT="${1:-8200}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_REPO="$SCRIPT_DIR/../external/openpi"
CHECKPOINT_DIR="${2:-gs://openpi-assets/checkpoints/pi05_libero}"

export OPENPI_DATA_HOME="/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/openpi"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
mkdir -p "$OPENPI_DATA_HOME"

python "${OPENPI_REPO}/scripts/serve_policy.py" \
    --port="${PORT}" \
    policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir="${CHECKPOINT_DIR}"
