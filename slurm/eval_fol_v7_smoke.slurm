#!/bin/bash
#SBATCH --job-name=fol_v7_smoke
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=5
#SBATCH --partition=GPU-shared --gres=gpu:v100-32:1
#SBATCH --time=01:30:00 --mem=63000M
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v7_smoke_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v7_smoke_%j.err

# FOL v7 — Real-World Compatible Multi-Prompt VLM Pipeline
# SMOKE TEST: safelibero_spatial Level I, n=2 per task
# Qwen2.5-VL-3B (6GB) + OpenVLA-7B (16GB) = ~22GB on V100-32
#
# v7 changes:
#   - 3-prompt design: obstacle ID, spatial predicates, physical properties
#   - Clean object names (no _obstacle suffix) passed to VLM
#   - VLM told explicitly which object is target/goal
#   - N=3 majority voting per prompt
#   - No MuJoCo GT information in filter core

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$REPO/.worktrees/fol-safety-filter"
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs "$WT/fol_v7_smoke" "$WT/fol_v7_smoke_video"

echo "=== FOL v7 Multi-Prompt SMOKE TEST safelibero_spatial Level I n=2 ===" && echo "Start: $(date)"
module load anaconda3 && conda activate openvla_libero_merged
export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

# ── Start Qwen2.5-VL-3B server ──
QWEN_LOG="$WT/fol_v7_smoke/qwen_server.log"
conda run --no-capture-output -n qwen \
    python "$REPO/vlm_pipeline/qwen_vlm_server.py" \
    --port 5001 --model qwen2.5-vl-3b \
    > "$QWEN_LOG" 2>&1 &
QWEN_PID=$!
echo "Qwen server PID: $QWEN_PID  (log: $QWEN_LOG)"

# Poll until ready (max 8 min = 96×5s)
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

# ── Verify /infer_v6 endpoint (used by VLM client) ──
V6_CHECK=$(curl -sf -X POST http://localhost:5001/infer_v6 \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Say ok","images_b64":[],"max_new_tokens":5}' 2>&1)
echo "infer_v6 health check: $V6_CHECK"

export QWEN_SERVER_URL=http://localhost:5001

python "$WT/vlm_pipeline/run_safelibero_fol_openvla_eval.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name safelibero_spatial --safety_level I \
    --num_trials_per_task 2 --use_fol_filter True --fol_level 1 \
    --results_output_dir "$WT/fol_v7_smoke" \
    --video_output_dir "$WT/fol_v7_smoke_video" \
    --center_crop True --seed 7 \
    --run_id_note "fol-v7-multiprompt-smoke"
EVAL_EXIT=$?

# ── Cleanup ──
echo "Shutting down Qwen server (PID=$QWEN_PID)"
kill "$QWEN_PID" 2>/dev/null
wait "$QWEN_PID" 2>/dev/null
echo "=== Done: $(date) (eval exit=$EVAL_EXIT) ==="
exit $EVAL_EXIT
