#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:2
#SBATCH --time=16:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=integrated_eval
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/slurm_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/slurm_%j.err

# =============================================================================
# SafeLIBERO Integrated Evaluation — OpenVLA-OFT + Qwen CBF Safety Filter
#
# Architecture:
#   GPU 0 — openvla_libero_merged: LIBERO sim + OpenVLA policy + CBF-QP
#   GPU 1 — qwen env: Qwen2.5-VL persistent HTTP server
#
# Per-chunk VLM calling: Qwen is called every num_open_loop_steps=8 steps
# (~38 calls/episode). Server keeps model loaded between calls (1-5s each).
#
# Usage:
#   sbatch slurm_integrated_eval.sh
#   # or override suite/level:
#   sbatch --export=SUITE=safelibero_object,LEVEL=II slurm_integrated_eval.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR="/ocean/projects/cis250185p/asingal"
QWEN_ENV="qwen"
OPENVLA_ENV="openvla_libero_merged"

SUITE="${SUITE:-safelibero_spatial}"
LEVEL="${LEVEL:-I}"
NUM_TRIALS="${NUM_TRIALS:-50}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10}"
VLM_MODEL="${VLM_MODEL:-qwen2.5-vl-7b}"
VLM_PORT="${VLM_PORT:-5001}"
VLM_VOTES="${VLM_VOTES:-3}"
VLM_TMP_DIR="${PROJECT_DIR}/tmp/vlm_live_${SLURM_JOB_ID}"
RESULTS_DIR="${PROJECT_DIR}/openvla_cbf_benchmark"
VIDEO_DIR="${PROJECT_DIR}/openvla_cbf_video"
LOG_DIR="${PROJECT_DIR}/slurm_logs"

module load anaconda3
mkdir -p "${LOG_DIR}" "${VLM_TMP_DIR}" "${RESULTS_DIR}" "${VIDEO_DIR}"

export MUJOCO_GL=egl
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export PYTHONPATH="${PROJECT_DIR}/openvla-oft:${PYTHONPATH:-}"
export PYTHONPATH="${PROJECT_DIR}/SafeLIBERO/safelibero:${PYTHONPATH}"

cd "${PROJECT_DIR}"

echo "Job ${SLURM_JOB_ID}: Suite=${SUITE} Level=${LEVEL} Trials=${NUM_TRIALS}"
echo "VLM model: ${VLM_MODEL} | Port: ${VLM_PORT} | Votes: ${VLM_VOTES}"
echo "VLM tmp dir: ${VLM_TMP_DIR}"

# ── Step 1: Start Qwen server on GPU 1 ───────────────────────────────────────
echo "Starting Qwen server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 conda run -n "${QWEN_ENV}" --no-capture-output \
    python -u "${PROJECT_DIR}/qwen_vlm_server.py" \
        --port "${VLM_PORT}" \
        --model "${VLM_MODEL}" \
    > "${LOG_DIR}/qwen_server_${SLURM_JOB_ID}.log" 2>&1 &
VLM_PID=$!
echo "Qwen server PID: ${VLM_PID}"
trap 'echo "Cleanup: stopping Qwen server (PID ${VLM_PID})"; kill "${VLM_PID}" 2>/dev/null || true; rm -rf "${VLM_TMP_DIR}" 2>/dev/null || true' SIGTERM SIGINT EXIT

# ── Step 2: Wait for server to load model (16 GB on Lustre: imports + weights ≈ 5-9 min) ────
echo "Waiting for Qwen server to load model..."
MAX_WAIT=600
if ! command -v curl &>/dev/null; then
    echo "ERROR: curl not found; cannot poll /health endpoint"
    exit 1
fi
for i in $(seq 1 ${MAX_WAIT}); do
    if curl -sf "http://localhost:${VLM_PORT}/health" > /dev/null 2>&1; then
        echo "Qwen server ready after ${i}s"
        break
    fi
    # Fail fast if the server process died
    if ! kill -0 "${VLM_PID}" 2>/dev/null; then
        echo "ERROR: Qwen server process (PID ${VLM_PID}) died — check ${LOG_DIR}/qwen_server_${SLURM_JOB_ID}.log"
        exit 1
    fi
    if [ "${i}" -eq "${MAX_WAIT}" ]; then
        echo "ERROR: Qwen server failed to start within ${MAX_WAIT}s"
        kill "${VLM_PID}" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# ── Step 3: Run evaluation on GPU 0 ──────────────────────────────────────────
echo "Starting evaluation on GPU 0..."
CUDA_VISIBLE_DEVICES=0 conda run -n "${OPENVLA_ENV}" --no-capture-output \
    python "${PROJECT_DIR}/run_libero_eval_integrated.py" \
        --task_suite_name "${SUITE}" \
        --safety_level "${LEVEL}" \
        --num_trials_per_task "${NUM_TRIALS}" \
        --use_safety_filter True \
        --vlm_dry_run False \
        --vlm_method m1 \
        --vlm_model "${VLM_MODEL}" \
        --num_vlm_votes "${VLM_VOTES}" \
        --vlm_server_url "http://localhost:${VLM_PORT}" \
        --vlm_tmp_dir "${VLM_TMP_DIR}" \
        --results_output_dir "${RESULTS_DIR}" \
        --video_output_dir "${VIDEO_DIR}" \
        --pretrained_checkpoint "${CHECKPOINT}" \
        --run_id_note "job${SLURM_JOB_ID}"
EVAL_EXIT=$?

# ── Step 4: Shut down Qwen server ────────────────────────────────────────────
# (handled by trap on EXIT)

# ── Step 5: Print results summary ────────────────────────────────────────────
echo ""
echo "=== Evaluation complete (exit code: ${EVAL_EXIT}) ==="
RESULTS_FILE=$(ls "${RESULTS_DIR}/${SUITE}/results_EVAL-"*"job${SLURM_JOB_ID}"*.json 2>/dev/null | head -1)
if [ -n "${RESULTS_FILE}" ]; then
    echo "Results: ${RESULTS_FILE}"
    RESULTS_FILE="${RESULTS_FILE}" conda run -n "${OPENVLA_ENV}" --no-capture-output python - <<'PYEOF'
import json, os, sys
path = os.environ["RESULTS_FILE"]
with open(path) as f:
    d = json.load(f)
o = d.get("overall", {})
print(f"  TSR={o.get('TSR','?')} CAR={o.get('CAR','?')} ETS_mean={o.get('ETS_mean','?')}")
print(f"  Total episodes: {o.get('total_episodes','?')}")
PYEOF
fi

exit "${EVAL_EXIT}"
