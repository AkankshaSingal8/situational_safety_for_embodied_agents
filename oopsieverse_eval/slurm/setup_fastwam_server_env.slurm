#!/bin/bash
#SBATCH --job-name=oopsieverse_fastwam_env_setup
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

module load anaconda3
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
bash /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval/oopsieverse_eval/setup/setup_fastwam_server_env.sh
