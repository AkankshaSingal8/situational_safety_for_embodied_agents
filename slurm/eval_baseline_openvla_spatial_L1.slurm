#!/bin/bash
#SBATCH --job-name=baseline_openvla_spatial_L1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=06:00:00
#SBATCH --mem=48GB
#SBATCH --output=logs/baseline_openvla_spatial_L1_%j.out
#SBATCH --error=logs/baseline_openvla_spatial_L1_%j.err

# Baseline: OpenVLA with NO safety filter on SafeLIBERO Spatial Level I
# This is the control experiment.

set -e
export MUJOCO_GL=egl

module load anaconda3
conda activate openvla_libero_merged

REPO_DIR="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/fol-safety-filter"
cd "$REPO_DIR/vlm_pipeline"

mkdir -p "$REPO_DIR/logs"

echo "=== Baseline (no filter) Evaluation ==="
echo "Job: $SLURM_JOB_ID  Start: $(date)"

python run_safelibero_fol_openvla_eval.py \
    --pretrained-checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task-suite-name safelibero_spatial \
    --safety-level I \
    --num-trials-per-task 10 \
    --results-output-dir "$REPO_DIR/baseline_benchmark" \
    --video-output-dir "$REPO_DIR/baseline_video" \
    --run-id-note "baseline-no-filter"

echo "=== Baseline complete: $(date) ==="
