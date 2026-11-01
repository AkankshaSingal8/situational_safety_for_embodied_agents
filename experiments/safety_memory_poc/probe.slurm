#!/bin/bash
#SBATCH --job-name=smpoc_probe
#SBATCH --account=cis240028p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=00:25:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
set -uo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
conda activate openvla_libero_merged
export MUJOCO_GL=egl
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safety_memory_poc
python probe_env.py
