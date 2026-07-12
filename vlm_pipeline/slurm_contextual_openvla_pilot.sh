#!/bin/bash
#SBATCH --job-name=ctx_openvla_pilot
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=62000M
#SBATCH --time=20:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate openvla_libero_merged

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE="$ROOT/.worktrees/contextual-predictive-safety-filter"
export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export PYTHONPATH=/ocean/projects/cis250185p/asingal/openvla-oft:$ROOT/SafeLIBERO/safelibero:$WORKTREE/vlm_pipeline:${PYTHONPATH:-}

TRIALS=${TRIALS:-50}
OUT=${OUT:-$WORKTREE/contextual_pilot/openvla}
for LEVEL in I II; do
  python "$WORKTREE/vlm_pipeline/run_safelibero_openvla_oft_eval.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name safelibero_spatial \
    --safety_level "$LEVEL" \
    --num_trials_per_task "$TRIALS" \
    --use_contextual_filter True \
    --run_id_note contextual-predictive \
    --video_output_dir "$OUT/videos" \
    --results_output_dir "$OUT/results"
done
