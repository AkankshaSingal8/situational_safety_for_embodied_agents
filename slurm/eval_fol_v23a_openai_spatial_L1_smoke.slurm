#!/bin/bash
#SBATCH --job-name=fol_v23a_openai_L1_smoke
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=5
#SBATCH --partition=GPU-shared --gres=gpu:v100-32:1
#SBATCH --time=03:00:00 --mem=63000M
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v23a_openai_spatial_L1_smoke_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v23a_openai_spatial_L1_smoke_%j.err

# v23a-openai — v22 stack (ellipsoid + multi-frame refine + hand checkpoint + ray-rescale
# + RGB-D) with the local Qwen VLM swapped for the OpenAI API backend via
# FOL_VLM_BACKEND=openai (Task A). No local Qwen server is started.
# SMOKE TEST: safelibero_spatial Level I, n=5 per task (20 total)

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$REPO/.worktrees/fol-safety-filter"
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs \
         "$WT/fol_v23a_openai_spatial_L1_smoke" "$WT/fol_v23a_openai_spatial_L1_smoke_video"

echo "=== FOL v23a (OpenAI VLM backend) spatial Level I smoke (n=5) ===" && echo "Start: $(date)"
module load anaconda3 && conda activate openvla_libero_merged
export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

set -a
source "$REPO/.env"
set +a
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY not set after sourcing $REPO/.env. Aborting."
    exit 1
fi

export FOL_VLM_BACKEND=openai
export FOL_VISION_GROUNDING=1
export FOL_ELLIPSOID=1
export FOL_REFINE_STEPS=2,6,10,14,18
export FOL_HAND_CHECKPOINT=1
export FOL_RAY_RESCALE=target
export FOL_RGBD=1

python "$WT/vlm_pipeline/run_safelibero_fol_openvla_eval.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name safelibero_spatial --safety_level I \
    --num_trials_per_task 5 --use_fol_filter True --fol_level 1 \
    --results_output_dir "$WT/fol_v23a_openai_spatial_L1_smoke" \
    --video_output_dir "$WT/fol_v23a_openai_spatial_L1_smoke_video" \
    --center_crop True --seed 7 \
    --run_id_note "fol-v23a-openai-spatial-L1-smoke"
EVAL_EXIT=$?

echo "=== Done: $(date) (eval exit=$EVAL_EXIT) ==="
exit $EVAL_EXIT
