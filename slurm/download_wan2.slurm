#!/bin/bash
#SBATCH --job-name=wan2_download
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/wan2_download_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/wan2_download_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /ocean/projects/cis250185p/asingal/envs/dreamzero_env

echo "Starting Wan2.1-I2V-14B-480P download at $(date)"
python /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/download_wan2.py
echo "Download complete at $(date)"
