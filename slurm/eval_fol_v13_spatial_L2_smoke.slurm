#!/bin/bash
#SBATCH --job-name=fol_v13_L2_smoke
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=5
#SBATCH --partition=GPU-shared --gres=gpu:v100-32:1
#SBATCH --time=03:00:00 --mem=63000M
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v13_spatial_L2_smoke_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v13_spatial_L2_smoke_%j.err

# FOL v13 — remove dist_ee_to_target<0.30 catch-22, h_cancel_override=0.25
# SMOKE TEST: safelibero_spatial Level II, n=5 per task (20 total)

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$REPO/.worktrees/fol-safety-filter"
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs \
         "$WT/fol_v13_spatial_L2_smoke" "$WT/fol_v13_spatial_L2_smoke_video"

echo "=== FOL v13 spatial Level II smoke (n=5) ===" && echo "Start: $(date)"
module load anaconda3 && conda activate openvla_libero_merged
export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

QWEN_LOG="$WT/fol_v13_spatial_L2_smoke/qwen_server.log"
conda run --no-capture-output -n qwen \
    python "$REPO/vlm_pipeline/qwen_vlm_server.py" \
    --port 5004 --model qwen2.5-vl-3b \
    > "$QWEN_LOG" 2>&1 &
QWEN_PID=$!
echo "Qwen server PID: $QWEN_PID  (log: $QWEN_LOG)"

echo "Waiting for Qwen server..."
READY=0
for i in $(seq 1 96); do
    if curl -sf http://localhost:5004/health > /dev/null 2>&1; then
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

export QWEN_SERVER_URL=http://localhost:5004

python "$WT/vlm_pipeline/run_safelibero_fol_openvla_eval.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name safelibero_spatial --safety_level II \
    --num_trials_per_task 5 --use_fol_filter True --fol_level 1 \
    --results_output_dir "$WT/fol_v13_spatial_L2_smoke" \
    --video_output_dir "$WT/fol_v13_spatial_L2_smoke_video" \
    --center_crop True --seed 7 \
    --run_id_note "fol-v13-spatial-L2-smoke"
EVAL_EXIT=$?

echo "Shutting down Qwen server (PID=$QWEN_PID)"
kill "$QWEN_PID" 2>/dev/null
wait "$QWEN_PID" 2>/dev/null
echo "=== Done: $(date) (eval exit=$EVAL_EXIT) ==="
exit $EVAL_EXIT
