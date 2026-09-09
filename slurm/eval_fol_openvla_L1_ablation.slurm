#!/bin/bash
#SBATCH --job-name=fol_ablation_L1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=10:00:00
#SBATCH --mem=48GB
#SBATCH --output=logs/fol_ablation_L1_%j.out
#SBATCH --error=logs/fol_ablation_L1_%j.err

# Ablation study: FOL Level 1 vs Level 2 vs Level 3 on Spatial Level I
# All three runs share the same GPU job.

set -e
export MUJOCO_GL=egl

module load anaconda3
conda activate openvla_libero_merged

REPO_DIR="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/fol-safety-filter"
cd "$REPO_DIR/vlm_pipeline"
mkdir -p "$REPO_DIR/logs"

echo "=== FOL Ablation Study: Level 1 ==="
python run_safelibero_fol_openvla_eval.py \
    --pretrained-checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task-suite-name safelibero_spatial --safety-level I \
    --num-trials-per-task 10 --use-fol-filter --fol-level 1 \
    --results-output-dir "$REPO_DIR/ablation_benchmark/level1" \
    --video-output-dir "$REPO_DIR/ablation_video/level1" \
    --run-id-note "fol-level1-spatial"

echo "=== FOL Ablation Study: Level 2 ==="
python run_safelibero_fol_openvla_eval.py \
    --pretrained-checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task-suite-name safelibero_spatial --safety-level I \
    --num-trials-per-task 10 --use-fol-filter --fol-level 2 \
    --results-output-dir "$REPO_DIR/ablation_benchmark/level2" \
    --video-output-dir "$REPO_DIR/ablation_video/level2" \
    --run-id-note "fol-level2-spatial"

echo "=== FOL Ablation Study: Level 3 ==="
python run_safelibero_fol_openvla_eval.py \
    --pretrained-checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task-suite-name safelibero_spatial --safety-level I \
    --num-trials-per-task 10 --use-fol-filter --fol-level 3 \
    --results-output-dir "$REPO_DIR/ablation_benchmark/level3" \
    --video-output-dir "$REPO_DIR/ablation_video/level3" \
    --run-id-note "fol-level3-spatial"

echo "=== Ablation complete: $(date) ==="
