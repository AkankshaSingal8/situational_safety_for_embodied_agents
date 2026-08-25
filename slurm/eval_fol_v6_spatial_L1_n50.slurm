#!/bin/bash
#SBATCH --job-name=fol_v6_spatial_L1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=5
#SBATCH --partition=GPU-shared --gres=gpu:v100-32:1
#SBATCH --time=12:00:00 --mem=63000M
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v6_spatial_L1_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v6_spatial_L1_%j.err

# FOL v6 — Multi-Prompt CoT with Composed Predicates
# FULL RUN: safelibero_spatial Level I, n=50 per task (200 episodes total)
# Uses Qwen2.5-VL-3B fp16 (~6GB) + OpenVLA-7B (~16GB) = ~22GB on V100-32
# 3B is faster to load (<5 min) and still capable for CoT scene reasoning

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$REPO/.worktrees/fol-safety-filter"
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs \
    "$WT/fol_v6_spatial_L1_n50" "$WT/fol_v6_spatial_L1_n50_video"

echo "=== FOL v6 CoT safelibero_spatial Level I n=50 ===" && echo "Start: $(date)"
module load anaconda3 && conda activate openvla_libero_merged
export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

# ── Start Qwen2.5-VL-3B (no quantization — fits on V100-32 with OpenVLA) ──
QWEN_LOG="$WT/fol_v6_spatial_L1_n50/qwen_server.log"
conda run --no-capture-output -n qwen \
    python "$REPO/vlm_pipeline/qwen_vlm_server.py" \
    --port 5001 --model qwen2.5-vl-3b \
    > "$QWEN_LOG" 2>&1 &
QWEN_PID=$!
echo "Qwen server PID: $QWEN_PID  (log: $QWEN_LOG)"

# Poll until server is ready (max 8 min = 96×5s; 3B loads in ~4 min)
echo "Waiting for Qwen server..."
READY=0
for i in $(seq 1 96); do
    if curl -sf http://localhost:5001/health > /dev/null 2>&1; then
        READY=1
        echo "Qwen server ready after ${i}x5s = $((i*5))s"
        break
    fi
    sleep 5
done

if [ "$READY" -eq 0 ]; then
    echo "ERROR: Qwen server did not start in 8 minutes. Aborting."
    kill "$QWEN_PID" 2>/dev/null
    exit 1
fi

# ── Verify /infer_v6 endpoint ──
V6_CHECK=$(curl -sf -X POST http://localhost:5001/infer_v6 \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Say ok","images_b64":[],"max_new_tokens":5}' 2>&1)
echo "infer_v6 health check: $V6_CHECK"

export QWEN_SERVER_URL=http://localhost:5001

python "$WT/vlm_pipeline/run_safelibero_fol_openvla_eval.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name safelibero_spatial --safety_level I \
    --num_trials_per_task 50 --use_fol_filter True --fol_level 1 \
    --fol_vlm_grounding True \
    --results_output_dir "$WT/fol_v6_spatial_L1_n50" \
    --video_output_dir "$WT/fol_v6_spatial_L1_n50_video" \
    --center_crop True --seed 7 \
    --run_id_note "fol-v6-cot-spatial-L1"
EVAL_EXIT=$?

# ── Cleanup ──
echo "Shutting down Qwen server (PID=$QWEN_PID)"
kill "$QWEN_PID" 2>/dev/null
wait "$QWEN_PID" 2>/dev/null
echo "=== Done: $(date) (eval exit=$EVAL_EXIT) ==="
exit $EVAL_EXIT
