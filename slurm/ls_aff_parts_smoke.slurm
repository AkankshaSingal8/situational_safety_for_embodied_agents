#!/bin/bash
#SBATCH --job-name=aff_parts
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=00:40:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Part-affordance grounding smoke (user-directed 2026-08-07): VLM part
# labels (committed d467987, N=5 unanimous) + GroundingDINO free-text part
# prompts vs the benchmark's GT part-geom centroids (goal
# Checkgrippercontactpart sets). No policy server. Gate: unsafe-part side
# discrimination >= 4/5 objects.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
conda activate openvla_libero_merged
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$LS/.libero_config"
export PYTHONPATH="$LS:$WT/client_deps:$WT/vlm_pipeline"
python "$WT/vlm_pipeline/smoke_affordance_parts.py"
echo "AFF_PARTS_SMOKE_DONE"
