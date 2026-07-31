#!/bin/bash
# CPU-only: adds openpi-client/websockets/imageio to the already-built
# oopsieverse_b1k container env, mirroring setup_robocasa_client_deps.slurm's
# approach exactly (RM-shared/low-qos CPU job for pip installs -- no GPU,
# no --nv needed, this is pure package installation). Run this BEFORE
# smoke_b1k_pi05.slurm / eval_pi05_fullsuite_b1k.slurm -- both scripts'
# run_b1k_pi05_eval.py imports openpi_client at module level and will fail
# immediately (after a ~25min Isaac Sim cold start) if this hasn't been run.
#SBATCH --job-name=oopsieverse_b1k_client_deps
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

EVAL_DIR=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval/oopsieverse_eval
SIF_PATH="$EVAL_DIR/setup/b1k_container.sif"

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

apptainer exec \
    --bind /ocean/projects/cis250185p/asingal:/ocean/projects/cis250185p/asingal \
    "$SIF_PATH" \
    bash "$EVAL_DIR/setup/setup_b1k_client_deps.sh"
