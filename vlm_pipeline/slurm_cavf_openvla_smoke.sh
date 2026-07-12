#!/bin/bash
# Additive CAVF proof-of-concept smoke test. Baselines and environments remain
# unchanged; output is isolated to the CAVF worktree.
#SBATCH --job-name=cavf_openvla_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

set -euo pipefail

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate openvla_libero_merged

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE="$ROOT/.worktrees/cavf-smoke-poc-20260712"
OUT="$WORKTREE/cavf_static_smoke/openvla"

mkdir -p "$OUT/results" "$OUT/videos"
export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export PYTHONPATH="$WORKTREE/vlm_pipeline:$ROOT/openvla-oft:$ROOT/SafeLIBERO/safelibero:\${PYTHONPATH:-}"

# Five matched initialization states for each of the four Spatial L1 tasks.
# This is a technical gate, not a publishable result.
python "$WORKTREE/vlm_pipeline/run_cavf_safelibero_openvla_eval.py" \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name safelibero_spatial \
  --safety_level I \
  --num_trials_per_task 5 \
  --use_contextual_filter True \
  --run_id_note cavf-static-smoke \
  --video_output_dir "$OUT/videos" \
  --results_output_dir "$OUT/results"
