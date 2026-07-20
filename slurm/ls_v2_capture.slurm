#!/bin/bash
#SBATCH --job-name=ls_v2cap
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --exclude=w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=01:30:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Scene-graph V2 prerequisite: capture 60 LIBERO-Safety scene snapshots
# (4 suites x 15 tasks) for offline graph-vs-flat hazard-ID validation.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
conda activate openvla_libero_merged
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$LS/.libero_config"
export PYTHONPATH="$LS:$WT/vlm_pipeline"
python "$WT/vlm_pipeline/libero_safety_capture.py"
