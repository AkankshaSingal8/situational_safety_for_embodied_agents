#!/bin/bash
#SBATCH --job-name=genesis_pilot_3a_ii
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:45:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
ENV_PATH=/ocean/projects/cis250185p/asingal/envs/genesis_env
OUTPUT_DIR=$REPO/genesis_bakeoff/scenario_3a_ii_output

mkdir -p "$OUTPUT_DIR"
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

module load cuda/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

pip install --quiet imageio 2>/dev/null || true

python "$REPO/genesis_bakeoff/metrics_3a_ii.py"

python "$REPO/genesis_bakeoff/scenario_3a_ii.py" \
    --output_dir "$OUTPUT_DIR" \
    --n_init_states 50 \
    --n_eval_episodes 10 \
    --n_unsafe_episodes 5
