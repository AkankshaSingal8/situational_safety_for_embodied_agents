#!/bin/bash
#SBATCH --job-name=fol_openvla_spatial_L1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=06:00:00
#SBATCH --mem=64000M
#SBATCH --output=logs/fol_openvla_spatial_L1_%j.out
#SBATCH --error=logs/fol_openvla_spatial_L1_%j.err

# FOL Safety Filter evaluation: OpenVLA + FOL Level 3 on SafeLIBERO Spatial Level I
#
# Results go to: fol_benchmark/safelibero_spatial/
# Compare against: openvla_benchmark/safelibero_spatial/ (baseline)

set -e
export MUJOCO_GL=egl

module load anaconda3
conda activate openvla_libero_merged

# Working directory
REPO_DIR="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/fol-safety-filter"
cd "$REPO_DIR/vlm_pipeline"

mkdir -p "$REPO_DIR/logs"

echo "=== FOL Safety Filter Evaluation ==="
echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"

python run_safelibero_fol_openvla_eval.py \
    --pretrained-checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task-suite-name safelibero_spatial \
    --safety-level I \
    --num-trials-per-task 10 \
    --use-fol-filter \
    --fol-level 3 \
    --fol-vlm-grounding False \
    --results-output-dir "$REPO_DIR/fol_benchmark" \
    --video-output-dir "$REPO_DIR/fol_video" \
    --run-id-note "fol-level3-heuristic"

echo "=== Evaluation complete: $(date) ==="
