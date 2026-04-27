#!/usr/bin/env bash
# vlm_prompt_runner/run_vlm.sh
#
# Convenience launcher for VLM inference on SafeLIBERO episodes.
# Activates the correct conda env and sets required env vars.
#
# Usage (interactive node or inside a SLURM job):
#   bash vlm_prompt_runner/run_vlm.sh \
#       --suite safelibero_spatial --level I --task 0 \
#       --prompt prompts/safety_predicates_prompt.md \
#       --vlm qwen2.5-vl-7b
#
# Environment variables:
#   CONDA_ENV     conda env to activate (default: qwen_vlm)
#   PROJECT_ROOT  project root path (default: parent of this script)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${PROJECT_ROOT:-$( dirname "$SCRIPT_DIR" )}"
CONDA_ENV="${CONDA_ENV:-qwen_vlm}"

# Activate conda if needed
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    echo "[run_vlm.sh] Activating conda env: $CONDA_ENV"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# Headless GPU rendering required on Bridges2 HPC
export MUJOCO_GL="${MUJOCO_GL:-egl}"

echo "[run_vlm.sh] PROJECT_ROOT=$PROJECT_ROOT"
echo "[run_vlm.sh] Args: $*"

cd "$PROJECT_ROOT"
python -m vlm_prompt_runner.main "$@"
