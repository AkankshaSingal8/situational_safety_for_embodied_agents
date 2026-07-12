#!/bin/bash
# CPU-only: just adds lightweight requests/json-numpy/openpi-client to the
# already-built oopsieverse_robocasa env. Run AFTER
# setup_pi05_server_env.slurm has cloned openpi.
#SBATCH --job-name=oopsieverse_robocasa_client_deps
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=00:20:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

module load anaconda3
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
bash /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval/oopsieverse_eval/setup/setup_robocasa_client_deps.sh
