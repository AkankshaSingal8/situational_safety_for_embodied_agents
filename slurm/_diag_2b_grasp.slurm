#!/bin/bash
#SBATCH --job-name=diag_2b_grasp
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:15:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
ENV_PATH=/ocean/projects/cis250185p/asingal/envs/genesis_env

module load cuda/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

python "$REPO/genesis_bakeoff/_diag_2b_grasp.py"
