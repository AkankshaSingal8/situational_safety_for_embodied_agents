#!/bin/bash
#SBATCH --job-name=genesis_1b_repeat
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
ENV_PATH=/ocean/projects/cis250185p/asingal/envs/genesis_env
OUTDIR=/ocean/projects/cis250185p/asingal/tmp/claude-101024/-ocean-projects-cis250185p-asingal-situational-safety-for-embodied-agents/4d544acd-7d8f-4cb8-9791-a1b46a35747a/scratchpad/1b_repeat_check
mkdir -p "$OUTDIR"
module load cuda/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"
pip install --quiet imageio 2>/dev/null || true
python "$REPO/genesis_bakeoff/scenario_1b.py" --output_dir "$OUTDIR" --n_init_states 50 --n_eval_episodes 10 --regenerate_init_states
