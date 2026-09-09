#!/bin/bash
#SBATCH --job-name=fol_v4_vlm_L1_n50
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=5
#SBATCH --partition=GPU-shared --gres=gpu:v100-32:1
#SBATCH --time=16:00:00 --mem=63000M
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v4_vlm_L1_n50_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v4_vlm_L1_n50_%j.err

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$REPO/.worktrees/fol-safety-filter"
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs "$WT/fol_v4_vlm_spatial_L1_n50"
echo "=== FOL v4 + Real Qwen VLM safelibero_spatial Level I n=50 ===" && echo "Start: $(date)"
module load anaconda3 && conda activate openvla_libero_merged
export MUJOCO_GL=egl HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

# ── Start Qwen server (7B 4-bit ~8GB; fits alongside OpenVLA-7B ~16GB on V100-32) ──
QWEN_LOG="$WT/fol_v4_vlm_spatial_L1_n50/qwen_server.log"
conda run --no-capture-output -n qwen \
    python "$REPO/vlm_pipeline/qwen_vlm_server.py" \
    --port 5001 --model qwen2.5-vl-7b --load_in_4bit \
    > "$QWEN_LOG" 2>&1 &
QWEN_PID=$!
echo "Qwen server PID: $QWEN_PID  (log: $QWEN_LOG)"

# Poll until server is ready (max 8 min for 7B model load)
echo "Waiting for Qwen server..."
READY=0
for i in $(seq 1 96); do
    if curl -sf http://localhost:5001/health > /dev/null 2>&1; then
        READY=1
        echo "Qwen server ready after ${i}x5s"
        break
    fi
    sleep 5
done

if [ "$READY" -eq 0 ]; then
    echo "ERROR: Qwen server did not start in 8 minutes. Aborting."
    kill "$QWEN_PID" 2>/dev/null
    exit 1
fi

export QWEN_SERVER_URL=http://localhost:5001
export PYTHONPATH="$WT/v4:$WT:$REPO/vlm_pipeline:$REPO/openvla-oft:$REPO/SafeLIBERO/safelibero:${PYTHONPATH:-}"
python "$WT/vlm_pipeline/run_safelibero_fol_openvla_eval.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name safelibero_spatial --safety_level I \
    --num_trials_per_task 50 --use_fol_filter True --fol_level 1 \
    --results_output_dir "$WT/fol_v4_vlm_spatial_L1_n50" \
    --video_output_dir "$WT/fol_v4_vlm_spatial_L1_n50_video" \
    --center_crop True --seed 7 --run_id_note "fol-v4-vlm-n50"
EVAL_EXIT=$?

# ── Cleanup ──
echo "Shutting down Qwen server (PID=$QWEN_PID)"
kill "$QWEN_PID" 2>/dev/null
wait "$QWEN_PID" 2>/dev/null
echo "=== Done: $(date) (eval exit=$EVAL_EXIT) ==="
exit $EVAL_EXIT
