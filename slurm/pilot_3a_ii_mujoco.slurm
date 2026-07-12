#!/bin/bash
# Bake-off pilot: scenario 3a-ii (open bag of Skittles) on MuJoCo/robosuite.
# Physics-only pilot (no policy loading) — abrupt bag-tip scatter, higher-energy
# variant of the 3a-i rice-pour pilot's controlled tilt-pour.
#SBATCH --job-name=pilot_3a_ii_mujoco
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /ocean/projects/cis250185p/asingal/envs/libero
export MUJOCO_GL=egl

python "$REPO/pilot_bakeoff/mujoco/skittles_pilot/run_skittles_scatter.py"

echo "=== Pilot 3a-ii (MuJoCo) done ==="
