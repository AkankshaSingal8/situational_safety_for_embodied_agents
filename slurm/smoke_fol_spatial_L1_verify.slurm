#!/bin/bash
# Phase 1/2 verification smoke: 1 task x 2 episodes, FOL filter on.
#
# Purpose: prove the fixed code in the code-review-fixes worktree runs
# end-to-end and produces a results JSON, before spending SU on any n=50 arm.
# Deliberately tiny -- 2 episodes, 1 task, 1 hour walltime.
#
# Paths are parameterized: WT holds OUR code (the worktree under test), REPO
# holds the populated submodules, which git worktrees do not inherit. Override
# either from the environment to point this at a different tree.
#SBATCH --job-name=smoke_fol_verify
#SBATCH --account=cis240028p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

REPO="${REPO:-/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents}"
WT="${WT:-$REPO/.worktrees/code-review-fixes}"
OUT="${OUT:-$WT/smoke_verify_out}"

echo "=== FOL smoke verification ==="
echo "REPO (submodules): $REPO"
echo "WT   (code under test): $WT"
echo "code commit: $(cd "$WT" && git rev-parse --short HEAD)"
echo "start: $(date)"

mkdir -p "$OUT"

module load anaconda3
conda activate /ocean/projects/cis250185p/asingal/envs/openvla_libero_merged

export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
# WT first so the worktree's fixed modules win over the main checkout's copies.
export PYTHONPATH="$WT:$WT/vlm_pipeline:$REPO/openvla-oft:$REPO/SafeLIBERO/safelibero:${PYTHONPATH:-}"

python "$WT/vlm_pipeline/run_safelibero_fol_openvla_eval.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 2 \
    --use_fol_filter True \
    --fol_level 1 \
    --results_output_dir "$OUT" \
    --center_crop True \
    --seed 7 \
    --run_id_note "smoke-verify-fixed-worktree"

echo "=== done: $(date) ==="
echo "--- results written ---"
find "$OUT" -name '*.json' -printf '%p (%s bytes)\n'
