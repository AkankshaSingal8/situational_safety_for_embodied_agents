#!/bin/bash
#SBATCH --job-name=dreamzero_bootstrap
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
ENV_PATH=/ocean/projects/cis250185p/asingal/envs/dreamzero_env
HF_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache
MERGED_CKPT=$HF_CACHE/hub/DreamZero-LIBERO-merged
DREAMZERO_DIR=$REPO/dream-policy/dreamzero
SERVER_PORT=8001
LOG_DIR=$REPO/dream_benchmark/logs

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p $LOG_DIR
mkdir -p $REPO/dream_benchmark

export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export MUJOCO_GL=egl

source "$(conda info --base)/etc/profile.d/conda.sh"

# ── Step 1: Create dreamzero_env if missing ───────────────────────────────────
if [ ! -d "$ENV_PATH" ]; then
    echo "=========================================="
    echo " STEP 1: Creating dreamzero_env (Python 3.11)"
    echo "=========================================="
    conda create -p "$ENV_PATH" python=3.11 -y
    conda activate "$ENV_PATH"

    echo "--- Installing PyTorch 2.8.0 + CUDA 12.4 ---"
    # Bridges2 H100 nodes run CUDA 12.4 driver-compatible builds
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu124

    echo "--- Installing DreamZero package dependencies ---"
    pip install \
        "transformers==4.51.3" \
        "diffusers==0.30.2" \
        "peft==0.5.0" \
        "accelerate>=1.0.0" \
        "safetensors>=0.5.0" \
        "einops>=0.8.0" \
        "hydra-core>=1.3.0" \
        "deepspeed>=0.16.0" \
        "flask>=3.0.0" \
        "flask-socketio>=5.5.0" \
        "python-socketio>=5.13.0" \
        "websockets>=13.0" \
        "huggingface-hub>=0.36.0" \
        "numpy==1.26.4" \
        "pillow>=11.0.0" \
        "opencv-python-headless>=4.11.0" \
        "imageio>=2.37.0" \
        "tqdm>=4.67.0" \
        "tyro" \
        "tianshou" \
        "openpi-client"

    echo "--- Installing DreamZero repo (editable) ---"
    pip install -e "$DREAMZERO_DIR"

    echo "--- Installing flash-attn (compiles against CUDA; ~10 min) ---"
    pip install flash-attn --no-build-isolation

    echo "--- Verifying GPU access ---"
    python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
print('CUDA version:', torch.version.cuda)
print('PyTorch version:', torch.__version__)
"
    echo "dreamzero_env created."
else
    echo "dreamzero_env already exists — skipping creation."
    conda activate "$ENV_PATH"
fi

conda activate "$ENV_PATH"

# ── Step 2: Wait for checkpoints (download runs on login node in background) ──
echo "=========================================="
echo " STEP 2: Checking checkpoints"
echo "=========================================="

BASE_CKPT=$HF_CACHE/hub/GEAR-Dreams/DreamZero-AgiBot
LORA_CKPT=$HF_CACHE/hub/KyleZ0906/DreamZero-LIBERO-LoRA

wait_for_file() {
    local path=$1
    local label=$2
    echo "Waiting for $label at $path ..."
    for i in $(seq 1 60); do
        if [ -d "$path" ] && [ "$(ls "$path"/*.safetensors "$path"/*.bin 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "  $label: found."
            return 0
        fi
        echo "  [$i/60] Not ready yet, sleeping 60s ..."
        sleep 60
    done
    echo "ERROR: $label not found after 60 minutes."
    exit 1
}

wait_for_file "$BASE_CKPT" "DreamZero-AgiBot base model"
wait_for_file "$LORA_CKPT" "DreamZero-LIBERO-LoRA adapter"

# ── Step 3: Merge LoRA into base model ───────────────────────────────────────
echo "=========================================="
echo " STEP 3: Merging LoRA into base model"
echo "=========================================="

if [ ! -d "$MERGED_CKPT" ] || [ -z "$(ls "$MERGED_CKPT" 2>/dev/null)" ]; then
    python "$REPO/dream-policy/merge_lora.py"
else
    echo "Merged checkpoint already exists at $MERGED_CKPT — skipping merge."
fi

ls -lh "$MERGED_CKPT"

# ── Step 4: Start DreamZero server in background ─────────────────────────────
echo "=========================================="
echo " STEP 4: Starting DreamZero server (port $SERVER_PORT)"
echo "=========================================="

cd "$DREAMZERO_DIR"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    socket_test_optimized_AR.py \
    --port "$SERVER_PORT" \
    --model-path "$MERGED_CKPT" \
    --enable-dit-cache \
    > "$LOG_DIR/server_bootstrap_${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# ── Step 5: Wait for server ready ────────────────────────────────────────────
echo "Waiting for DreamZero server (up to 10 minutes)..."
for i in $(seq 1 120); do
    sleep 5
    if grep -qEi "listening|running|ready|started|serving|8000|$SERVER_PORT" \
            "$LOG_DIR/server_bootstrap_${SLURM_JOB_ID}.log" 2>/dev/null; then
        echo "Server ready after $((i*5))s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died. Last 20 lines of log:"
        tail -20 "$LOG_DIR/server_bootstrap_${SLURM_JOB_ID}.log"
        exit 1
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: Server did not start in 10 minutes."
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
done

# ── Step 6: Smoke test (1 episode, task 0) ───────────────────────────────────
echo "=========================================="
echo " STEP 6: Smoke test (1 episode, task 0)"
echo "=========================================="

conda activate libero_env
export MUJOCO_GL=egl
export PYTHONPATH=$REPO/vlm_pipeline:$REPO/SafeLIBERO/safelibero:${PYTHONPATH:-}

# Install WebSocket + client deps into libero_env if missing
pip install --quiet websockets>=13.0 opencv-python-headless openpi-client 2>/dev/null || true

python "$REPO/vlm_pipeline/run_safelibero_dreamzero_eval.py" \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 1 \
    --task_indices 0 \
    --server_host localhost \
    --server_port "$SERVER_PORT" \
    --replan_steps 24 \
    --results_output_dir /tmp/dreamzero_smoke \
    --save_videos

EVAL_EXIT=$?

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo "Stopping server (PID $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo " Smoke test result (exit code: $EVAL_EXIT)"
echo "=========================================="
find /tmp/dreamzero_smoke -name "*.json" 2>/dev/null | head -1 | xargs python -m json.tool 2>/dev/null || \
    find /tmp/dreamzero_smoke -name "*.json" 2>/dev/null | xargs cat 2>/dev/null || \
    echo "No result JSON found"

exit $EVAL_EXIT
