#!/bin/bash
#SBATCH --job-name=cavf_calibrate
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=00:20:00

set -euo pipefail

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE="$ROOT/.worktrees/cavf-smoke-poc-20260712"
RUN_ROOT="$WORKTREE/cavf_gates/controller_calibration"
mkdir -p "$RUN_ROOT/logs"
exec > >(tee -a "$RUN_ROOT/logs/driver_$SLURM_JOB_ID.log") 2>&1

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate openvla_libero_merged

export MUJOCO_GL=egl
export PYTHONPATH="$WORKTREE/vlm_pipeline:$ROOT/SafeLIBERO/safelibero:\${PYTHONPATH:-}"

python -u "$WORKTREE/vlm_pipeline/calibrate_safelibero_controller.py" \
  --suite safelibero_spatial \
  --level I \
  --task-id 0 \
  --state-count 5 \
  --output "$RUN_ROOT/controller_response.json"
