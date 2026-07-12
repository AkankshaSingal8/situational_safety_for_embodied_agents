#!/bin/bash
#SBATCH --job-name=oopsieverse_pi05_env_setup
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=01:30:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

module load anaconda3
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
bash /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval/oopsieverse_eval/setup/setup_pi05_server_env.sh
