#!/bin/bash
#SBATCH --job-name=dreamzero_bootstrap
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
ENV_PATH=/ocean/projects/cis250185p/asingal/envs/dreamzero_env
HF_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache
# LoRA checkpoint dir — VLA.load_lora() loads directly from here (no merge needed).
LORA_CKPT=$HF_CACHE/hub/KyleZ0906/DreamZero-LIBERO-LoRA
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

    echo "--- Installing PyTorch 2.8.0 + CUDA 12.6 ---"
    # Bridges2 default CUDA module is 12.6.1 → use cu126 wheels
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu126

    echo "--- Installing DreamZero server dependencies ---"
    # Install only the subset needed to run socket_test_optimized_AR.py.
    # Full pyproject.toml has gear/tensorrt/nvidia-modelopt/pin/decord2 which are
    # NVIDIA-internal or unavailable on PyPI — skip those via --no-deps below.
    pip install \
        "transformers==4.51.3" \
        "diffusers==0.30.2" \
        "peft>=0.10.0" \
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
        "imageio==2.34.2" \
        "imageio-ffmpeg" \
        "av" \
        "tqdm>=4.67.0" \
        "tyro" \
        "tianshou==0.5.1" \
        "openpi-client==0.1.1" \
        "msgpack" \
        "msgpack-numpy" \
        "scipy>=1.15.0" \
        "click" \
        "termcolor" \
        "loguru" \
        "pyzmq" \
        "tiktoken" \
        "sentencepiece" \
        "datasets==3.6.0" \
        "wandb" \
        "dm_tree" \
        "albumentations==1.4.18"

    echo "--- Installing DreamZero repo (editable, no-deps to skip unavailable NVIDIA packages) ---"
    pip install -e "$DREAMZERO_DIR" --no-deps

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

# ── Step 2: Wait for LoRA checkpoint (download runs on login node) ────────────
echo "=========================================="
echo " STEP 2: Checking LoRA checkpoint"
echo "=========================================="

# Only the LoRA adapter is needed. VLA.load_lora() reads model.safetensors from
# LORA_CKPT and auto-downloads Wan2.1 sub-components from HuggingFace if missing.
LORA_WEIGHTS="$LORA_CKPT/model.safetensors"
LORA_EXPECTED_BYTES=217288600
echo "Waiting for LoRA weights at $LORA_WEIGHTS (expected ${LORA_EXPECTED_BYTES} bytes) ..."
for i in $(seq 1 180); do
    # Use python for reliable size — Lustre can lag on stat/ls metadata
    ACTUAL=$(python3 -c "import os; print(os.path.getsize('$LORA_WEIGHTS'))" 2>/dev/null || echo 0)
    if [ "${ACTUAL}" -ge "${LORA_EXPECTED_BYTES}" ]; then
        echo "  LoRA model.safetensors: complete (${ACTUAL} bytes)."
        break
    fi
    echo "  [$i/180] Not ready yet (${ACTUAL}/${LORA_EXPECTED_BYTES} bytes), sleeping 60s ..."
    sleep 60
    if [ $i -eq 180 ]; then
        echo "ERROR: LoRA model.safetensors not complete after 180 minutes."
        exit 1
    fi
done

ls -lh "$LORA_CKPT"

# ── Step 3: Start DreamZero server in background ─────────────────────────────
echo "=========================================="
echo " STEP 3: Starting DreamZero server (port $SERVER_PORT)"
echo "=========================================="

# NOTE: --model-path is the LoRA adapter directory. GrootSimPolicy reads
# experiment_cfg/conf.yaml (save_lora_only: true) and calls VLA.load_lora(),
# which auto-downloads Wan2.1-I2V-14B-480P (~40GB) on first run.
# First startup may take 30-60 minutes.
# ATTENTION_BACKEND=torch: flash-attn is not installed (requires nvcc to build);
# wan2_1_attention.py falls back to PyTorch sdp when env var is set to "torch".
export ATTENTION_BACKEND=torch
export NO_ALBUMENTATIONS_UPDATE=1

cd "$DREAMZERO_DIR"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    socket_test_optimized_AR.py \
    --port "$SERVER_PORT" \
    --model-path "$LORA_CKPT" \
    --enable-dit-cache \
    > "$LOG_DIR/server_bootstrap_${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# ── Step 4: Wait for server ready ────────────────────────────────────────────
# First run downloads Wan2.1-I2V-14B-480P (~40GB); allow up to 90 minutes.
echo "Waiting for DreamZero server (up to 90 minutes — first run downloads Wan2.1)..."
for i in $(seq 1 1080); do
    sleep 5
    # websockets library logs "server listening on 0.0.0.0:PORT" — match this
    # specifically to avoid false positives from traceback text containing "run/running".
    if grep -qEi "listening on|server started on" \
            "$LOG_DIR/server_bootstrap_${SLURM_JOB_ID}.log" 2>/dev/null; then
        echo "Server ready after $((i*5))s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died. Last 30 lines of log:"
        tail -30 "$LOG_DIR/server_bootstrap_${SLURM_JOB_ID}.log"
        exit 1
    fi
    if [ $((i % 60)) -eq 0 ]; then
        echo "Still waiting ($((i*5))s elapsed). Last server log line:"
        tail -1 "$LOG_DIR/server_bootstrap_${SLURM_JOB_ID}.log" 2>/dev/null || echo "(no log yet)"
    fi
    if [ $i -eq 1080 ]; then
        echo "ERROR: Server did not start in 90 minutes. Last 30 lines:"
        tail -30 "$LOG_DIR/server_bootstrap_${SLURM_JOB_ID}.log"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
done

# ── Step 5: Smoke test (1 episode, task 0) ───────────────────────────────────
echo "=========================================="
echo " STEP 5: Smoke test (1 episode, task 0)"
echo "=========================================="

conda activate /ocean/projects/cis250185p/asingal/envs/libero
export MUJOCO_GL=egl
export PYTHONPATH=$REPO/vlm_pipeline:$REPO/SafeLIBERO/safelibero:${PYTHONPATH:-}

# Install WebSocket + client deps into libero env if missing
pip install --quiet "websockets>=13.0" opencv-python-headless openpi-client 2>/dev/null || true

python "$REPO/vlm_pipeline/run_safelibero_dreamzero_eval.py" \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 1 \
    --task_indices 0 \
    --server_host localhost \
    --server_port "$SERVER_PORT" \
    --replan_steps 24 \
    --results_output_dir /tmp/dreamzero_smoke \
    --video_output_dir "$REPO/dream_benchmark/videos" \
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
