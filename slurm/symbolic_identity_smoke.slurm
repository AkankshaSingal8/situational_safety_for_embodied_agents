#!/bin/bash
#SBATCH --job-name=ident_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --exclude=w005,w006,w007,w008,w009
#SBATCH --time=01:30:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Live-sim smoke for runtime RGB-D obstacle localization (no-GT tier geometry leg).
set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
conda activate openvla_libero_merged
export MUJOCO_GL=egl
export PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/vlm_pipeline"
python "$WT/vlm_pipeline/symbolic_identity_smoke.py"
