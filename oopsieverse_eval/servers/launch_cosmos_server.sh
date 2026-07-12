#!/usr/bin/env bash
# Launches cosmos_server.py inside cosmos-policy's existing Apptainer
# container + .venv_rhel8 -- read-only reuse, no rebuild. Mirrors the exact
# working apptainer invocation from
# slurm/eval_cosmos_safelibero_spatial_L1.slurm (env vars + binds only, no
# extra host-lib overrides -- see cosmos_risk_gate_spike.py's docstring for
# why that matters).
#
# Usage: bash launch_cosmos_server.sh [port]
set -euo pipefail

PORT="${1:-8400}"
REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
COSMOS_DIR="$REPO/cosmos-policy"
SIF_PATH="$COSMOS_DIR/cosmos_policy.sif"
SERVER_SCRIPT="$REPO/.worktrees/oopsieverse-eval/oopsieverse_eval/servers/cosmos_server.py"

if [ ! -f "$SIF_PATH" ]; then
  echo "ERROR: SIF not found at $SIF_PATH -- this reuses the existing container read-only."
  exit 1
fi

apptainer exec --nv \
    --env MUJOCO_GL=egl \
    --env HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache \
    --env TRANSFORMERS_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache \
    --env HF_TOKEN=$(cat /jet/home/asingal/.cache/huggingface/token 2>/dev/null || echo "") \
    --bind $REPO:$REPO \
    --bind /ocean/projects/cis250185p/asingal/.hf_cache:/ocean/projects/cis250185p/asingal/.hf_cache \
    --bind /jet/home/asingal:/jet/home/asingal \
    "$SIF_PATH" \
    bash -c "$COSMOS_DIR/.venv_rhel8/bin/python $SERVER_SCRIPT --port $PORT"
