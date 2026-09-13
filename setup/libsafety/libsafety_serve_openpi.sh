#!/usr/bin/env bash
set -euo pipefail
# Usage: bash libsafety_serve_openpi.sh <pi0_libero|pi05_libero> [port]
#
# Starts the openpi policy server for pi0 / pi0.5 LIBERO-Safety checkpoints.
# Must run inside the libsafety_openpi conda env (python 3.11, Task 3).
#
# NOTE on openpi location: vlsa-aegis/openpi's editable install in
# libsafety_openpi resolves to the MAIN repo's copy
# (.../situational_safety_for_embodied_agents/vlsa-aegis/openpi), confirmed
# via `pip show openpi` -> "Editable project location: .../vlsa-aegis/openpi"
# (no .worktrees/ in the path), NOT this worktree's copy. Both copies happen
# to be populated and at the same git commit, but we use the main-repo path
# here to match what's actually importable in this env.
POLICY_CONFIG="${1:?pass pi0_libero or pi05_libero}"
PORT="${2:-8000}"
MAIN_REPO="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents"
LIBSAFETY_DIR="${MAIN_REPO}/.worktrees/libero-safety-benchmark/LIBERO-Safety"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openpi

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# NOTE: tyro parses --port as a top-level option, which must precede the
# "policy:checkpoint" subcommand token; passing it after the subcommand's
# own --policy.* flags raises "Unrecognized options: --port=...".
python "${MAIN_REPO}/vlsa-aegis/openpi/scripts/serve_policy.py" \
    --port="${PORT}" \
    policy:checkpoint \
    --policy.config="${POLICY_CONFIG}" \
    --policy.dir="${LIBSAFETY_DIR}/checkpoints/${POLICY_CONFIG}"
