#!/bin/bash
#SBATCH --job-name=dreamzero_spatial_L1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
HF_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache
MERGED_CKPT=$HF_CACHE/hub/DreamZero-LIBERO-merged
SERVER_PORT=8000
LOG_DIR=$REPO/dream_benchmark/logs

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p $LOG_DIR
mkdir -p $REPO/dream_benchmark
mkdir -p $REPO/dream_video

source "$(conda info --base)/etc/profile.d/conda.sh"

# ── Pre-flight check ──────────────────────────────────────────────────────────
if [ ! -d "$MERGED_CKPT" ]; then
    echo "[$(date)] ERROR: Merged checkpoint not found at $MERGED_CKPT"
    echo "Run first: conda activate /ocean/projects/cis250185p/asingal/envs/dreamzero_env"
    echo "           python $REPO/dream-policy/merge_lora.py"
    exit 1
fi

if [ ! -f "$REPO/dream-policy/dreamzero/socket_test_optimized_AR.py" ]; then
    echo "[$(date)] ERROR: DreamZero server not cloned."
    echo "Run first: git clone https://github.com/dreamzero0/dreamzero $REPO/dream-policy/dreamzero"
    exit 1
fi

# ── 1. Start DreamZero server in background ───────────────────────────────────
echo "[$(date)] Starting DreamZero server on port $SERVER_PORT..."
conda activate /ocean/projects/cis250185p/asingal/envs/dreamzero_env
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export MUJOCO_GL=egl

cd $REPO/dream-policy/dreamzero
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    socket_test_optimized_AR.py \
    --port $SERVER_PORT \
    --model-path $MERGED_CKPT \
    --enable-dit-cache \
    > $LOG_DIR/server_${SLURM_JOB_ID}.log 2>&1 &

SERVER_PID=$!
echo "[$(date)] Server PID: $SERVER_PID"

# ── 2. Wait for server to become ready (up to 10 minutes) ────────────────────
echo "[$(date)] Waiting for DreamZero server to start (up to 10 min)..."
for i in $(seq 1 120); do
    sleep 5
    if grep -qEi "listening|running|ready|started|serving" \
            "$LOG_DIR/server_${SLURM_JOB_ID}.log" 2>/dev/null; then
        echo "[$(date)] Server ready after $((i * 5))s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[$(date)] ERROR: Server process died. Check log:"
        tail -20 "$LOG_DIR/server_${SLURM_JOB_ID}.log"
        exit 1
    fi
    if [ $i -eq 120 ]; then
        echo "[$(date)] ERROR: Server did not start after 10 minutes."
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
done

# ── 3. Run evaluation client ──────────────────────────────────────────────────
echo "[$(date)] Starting SafeLIBERO evaluation..."
conda activate libero_env
export MUJOCO_GL=egl
export PYTHONPATH=$REPO/vlm_pipeline:$REPO/SafeLIBERO/safelibero:${PYTHONPATH:-}

python $REPO/vlm_pipeline/run_safelibero_dreamzero_eval.py \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 10 \
    --server_host localhost \
    --server_port $SERVER_PORT \
    --replan_steps 24 \
    --results_output_dir $REPO/dream_benchmark \
    --video_output_dir $REPO/dream_video \
    --seed 195

EVAL_EXIT=$?

# ── 4. Cleanup ────────────────────────────────────────────────────────────────
echo "[$(date)] Evaluation done (exit code: $EVAL_EXIT). Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "[$(date)] Job complete."
exit $EVAL_EXIT
