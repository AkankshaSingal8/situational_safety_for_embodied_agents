#!/bin/bash
#SBATCH --job-name=diag_3a_ii
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
ENV_PATH=/ocean/projects/cis250185p/asingal/envs/genesis_env

module load cuda/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

# NOTE: gs.init() is process-global (can only be called once per process,
# see _diag_3a_ii.py docstring) -- run each config as its own subprocess
# invocation instead of looping in-process.
run() {
  python "$REPO/genesis_bakeoff/_diag_3a_ii.py" --pack_spacing "$1" --dt "$2" --substeps "$3" --seed "$4"
}

for seed in 0 1 2; do
  run 0.012 0.005 20 $seed   # original pilot params
  run 0.0145 0.005 20 $seed  # reduced overlap (near-touching)
  run 0.012 0.005 40 $seed   # original overlap, finer substeps
  run 0.012 0.002 20 $seed   # original overlap, smaller dt
done

echo "ALL_DIAG_3A_II_CONFIGS_DONE"
