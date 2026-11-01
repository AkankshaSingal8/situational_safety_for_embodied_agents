#!/bin/bash
#SBATCH --job-name=smpoc_verify
#SBATCH --account=cis240028p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=00:20:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
set -uo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
conda activate openvla_libero_merged
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safety_memory_poc
python verify_figure.py
