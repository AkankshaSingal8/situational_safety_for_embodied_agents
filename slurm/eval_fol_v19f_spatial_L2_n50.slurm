#!/bin/bash
#SBATCH --job-name=fol_v19f_L2
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=5
#SBATCH --partition=GPU-shared --gres=gpu:v100-32:1
#SBATCH --time=16:00:00 --mem=63000M
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v19f_spatial_L2_n50_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v19f_spatial_L2_n50_%j.err

# FOL v19 — GT-free + covariance-aligned ellipsoid barrier, image-derived extents, multi-frame wrist refinement, step-10 re-ground retry
# wrist+elbow checkpoints only), keep v6 VLM/FOL grounding pipeline, SSF disabled.
# v3 achieved TSR=48.5% CAR=59.5% on spatial L1; v7-v14 cylindrical additions
# were fixes for phantom problems (filter was dead from NameError) and crushed TSR.
# SMOKE TEST: safelibero_spatial Level I, n=5 per task (20 total)

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$REPO/.worktrees/fol-safety-filter"
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs \
         "$WT/fol_v19f_spatial_L2_n50" "$WT/fol_v19f_spatial_L2_n50_video"

echo "=== FOL v19 spatial Level II n=50 ===" && echo "Start: $(date)"
module load anaconda3 && conda activate openvla_libero_merged
export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

QWEN_LOG="$WT/fol_v19f_spatial_L2_n50/qwen_server.log"
conda run --no-capture-output -n qwen \
    python "$REPO/vlm_pipeline/qwen_vlm_server.py" \
    --port 5036 --model qwen2.5-vl-3b \
    > "$QWEN_LOG" 2>&1 &
QWEN_PID=$!
echo "Qwen server PID: $QWEN_PID  (log: $QWEN_LOG)"

echo "Waiting for Qwen server..."
READY=0
for i in $(seq 1 300); do
    if curl -sf http://localhost:5036/health > /dev/null 2>&1; then
        READY=1
        echo "Qwen server ready after ${i}x5s = $((i*5))s"
        break
    fi
    sleep 5
done

if [ "$READY" -eq 0 ]; then
    echo "ERROR: Qwen server did not start. Aborting."
    kill "$QWEN_PID" 2>/dev/null
    exit 1
fi

export QWEN_SERVER_URL=http://localhost:5036
export FOL_VISION_GROUNDING=1
export FOL_ELLIPSOID=1
export FOL_REFINE_STEPS=6,12,18

python "$WT/vlm_pipeline/run_safelibero_fol_openvla_eval.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name safelibero_spatial --safety_level II \
    --num_trials_per_task 50 --use_fol_filter True --fol_level 1 \
    --results_output_dir "$WT/fol_v19f_spatial_L2_n50" \
    --video_output_dir "$WT/fol_v19f_spatial_L2_n50_video" \
    --center_crop True --seed 7 \
    --run_id_note "fol-v19f-spatial-L2"
EVAL_EXIT=$?

echo "Shutting down Qwen server (PID=$QWEN_PID)"
kill "$QWEN_PID" 2>/dev/null
wait "$QWEN_PID" 2>/dev/null
echo "=== Done: $(date) (eval exit=$EVAL_EXIT) ==="
exit $EVAL_EXIT
