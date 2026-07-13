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

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE="$ROOT/.worktrees/cavf-smoke-poc-20260712"
OUT="$WORKTREE/cavf_static_smoke/openvla"
RUN_ROOT="$WORKTREE/cavf_static_smoke"
DRIVER_LOG="$RUN_ROOT/logs/driver_$SLURM_JOB_ID.log"

mkdir -p "$OUT/results" "$OUT/videos" "$RUN_ROOT/logs"
exec > >(tee -a "$DRIVER_LOG") 2>&1

on_exit() {
  status=$?
  printf '\n=== CAVF OpenVLA smoke exit status: %s ===\n' "$status"
  printf 'Driver log: %s\n' "$DRIVER_LOG"
  printf 'Results directory: %s\n' "$OUT/results"
  exit "$status"
}
trap on_exit EXIT

printf '=== CAVF OpenVLA smoke ===\n'
printf 'job_id=%s host=%s started=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
printf 'worktree=%s\n' "$WORKTREE"
git -C "$WORKTREE" rev-parse --abbrev-ref HEAD
git -C "$WORKTREE" rev-parse --short HEAD

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate openvla_libero_merged
printf 'python=%s\n' "$(command -v python)"
python --version

export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export PYTHONPATH="$WORKTREE/vlm_pipeline:$ROOT/openvla-oft:$ROOT/SafeLIBERO/safelibero:\${PYTHONPATH:-}"

# Five matched initialization states for each of the four Spatial L1 tasks.
# This is a technical gate, not a publishable result.
python -u "$WORKTREE/vlm_pipeline/run_cavf_safelibero_openvla_eval.py" \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name safelibero_spatial \
  --safety_level I \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK:-5}" \
  --use_contextual_filter True \
  --run_id_note cavf-static-smoke \
  --video_output_dir "$OUT/videos" \
  --results_output_dir "$OUT/results"

printf '\n=== Result artifacts ===\n'
find "$OUT/results" -maxdepth 2 -type f -name '*.json' -print
