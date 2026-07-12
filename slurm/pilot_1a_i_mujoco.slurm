#!/bin/bash
# Bake-off pilot: scenario 1a-i (mug behind tall bottle) on MuJoCo/robosuite.
# Validates the new BDDL + .pruned_init files run end-to-end through
# OpenVLA-OFT policy inference and the run_episode() step loop.
#SBATCH --job-name=pilot_1a_i_mujoco
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p "$REPO/pilot_bakeoff/mujoco/videos"
mkdir -p "$REPO/pilot_bakeoff/mujoco/results"

module load anaconda3
conda activate openvla_libero_merged

export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache
export PYTHONPATH="$REPO/vlm_pipeline:$REPO/openvla-oft:$REPO/SafeLIBERO/safelibero:${PYTHONPATH:-}"

python "$REPO/vlm_pipeline/run_pilot_1a_i.py" \
    --pretrained_checkpoint "moojink/openvla-7b-oft-finetuned-libero-spatial" \
    --num_episodes 2 \
    --vlm_dry_run True \
    --results_output_dir "$REPO/pilot_bakeoff/mujoco/results" \
    --video_output_dir "$REPO/pilot_bakeoff/mujoco/videos" \
    --local_log_dir "$REPO/pilot_bakeoff/mujoco/logs" \
    --center_crop True \
    --seed 7

echo "=== Pilot 1a-i (MuJoCo) done ==="
