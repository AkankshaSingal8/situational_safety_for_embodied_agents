#!/bin/bash
# pi0.5 (off-the-shelf, pi05_libero) smoke test on OopsieVerse BEHAVIOR-1K
# (OmniGibson): 3 episodes each on 3 representative tasks (pick_egg as a
# baseline sanity task; add_firewood + wipe_counter as 2 of the paper's
# matched 4-task subset -- see run_b1k_pi05_eval.py's PAPER_SUBSET_TASKS
# comment for the full name-matching reasoning), to validate the
# server + client + adapter + action-space-conversion wiring before
# scaling to the full 13-task suite.
#
# SLURM ARRAY (one array index per task), mirroring
# eval_pi05_fullsuite_b1k.slurm's shape -- NOT one process looping over
# all 3 tasks. `B1KAdapter.close()` calls `og.shutdown()`, which tears
# down the whole Kit SimulationApp; constructing a second
# `OGDamageableEnvironment` afterwards in the same process is not a
# supported path (and OmniGibson's own `shutdown()` may itself call
# `exit()`, in which case a looping-over-tasks single process would
# silently lose every task's results after the first). Three independent
# array shards cost the same wall-clock as one (they run concurrently,
# cluster availability permitting) and each writes to its own results
# subdir -- see eval_pi05_fullsuite_b1k.slurm's own comment on why shards
# racing on one shared episodes.json silently keep only 1/N of the results.
#
# Requests 2 GPUs per shard, NOT 1: Isaac Sim's RTX renderer needs real
# GPU device creation (`[gpu.foundation.plugin] No device could be created`
# was one of the five real failure modes hit getting the headless spike
# working -- see task-1-report.md), and `launch_pi05_server.sh` sets
# `XLA_PYTHON_CLIENT_MEM_FRACTION` (lowered to 0.4 here, from RoboCasa's
# 0.9, since Isaac Sim's Vulkan device enumeration may not honor
# CUDA_VISIBLE_DEVICES the way CUDA/JAX do -- cheap headroom either way).
#
# Runs the client INSIDE b1k_container.sif (mandatory -- see
# spike_b1k_headless.slurm / task-1-report.md for why bare metal cannot
# run Isaac Sim on this cluster at all), with the exact same
# LD_LIBRARY_PATH additions + Vulkan ICD binds that spike needed.
#SBATCH --job-name=oopsieverse_b1k_pi05_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96000M
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail
echo "=== OopsieVerse BEHAVIOR-1K x pi0.5 smoke test (array) ==="
echo "Host: $(hostname)"
nvidia-smi || echo "WARNING: nvidia-smi failed"

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
EVAL_DIR="$REPO/.worktrees/oopsieverse-eval/oopsieverse_eval"
SIF_PATH="$EVAL_DIR/setup/b1k_container.sif"
PORT=$((8300 + SLURM_ARRAY_TASK_ID))
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/pi05_server_b1k_smoke_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

TASKS=(pick_egg add_firewood wipe_counter)
TASK_NAME="${TASKS[$SLURM_ARRAY_TASK_ID]}"
# Only the first shard runs the policy-free controller/motion/gripper
# diagnostic (identical env construction across shards, so one is enough;
# running it 3x would just waste cold-start time).
DIAGNOSE_FLAG=""
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    DIAGNOSE_FLAG="--diagnose"
fi
echo "=== task=$TASK_NAME (array idx $SLURM_ARRAY_TASK_ID), port=$PORT, diagnose=${DIAGNOSE_FLAG:-no} ==="

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

if [ ! -f "$SIF_PATH" ]; then
    echo "ERROR: SIF not found at $SIF_PATH -- build it first with"
    echo "  sbatch $EVAL_DIR/slurm/build_b1k_sif.slurm"
    exit 1
fi

module load anaconda3

# ── Start pi0.5 server on GPU 0, unchanged (reuses the exact RoboCasa server) ──
conda activate /ocean/projects/cis250185p/asingal/envs/oopsieverse_pi05_srv
export OPENPI_DATA_HOME=/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export CUDA_VISIBLE_DEVICES=0

echo "Starting pi0.5 server on GPU 0..."
bash "$EVAL_DIR/servers/launch_pi05_server.sh" "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

cleanup() {
    echo "Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server on port $PORT..."
MAX_WAIT=1800
ELAPSED=0
until nc -z localhost "$PORT" 2>/dev/null; do
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo "  ...${ELAPSED}s"
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Server did not start. Log:"
        cat "$SERVER_LOG"
        exit 1
    fi
done
echo "Server ready!"

# ── Run eval client on GPU 1, inside the B1K Apptainer container ────────
LD_LIBRARY_PATH_ADDITIONS_CMD='ENVROOT=/ocean/projects/cis250185p/asingal/envs/oopsieverse_b1k/lib/python3.10/site-packages/isaacsim; LDP=$(find "$ENVROOT" -maxdepth 3 -type d -name bin 2>/dev/null | tr "\n" ":"); export LD_LIBRARY_PATH="$LDP:${LD_LIBRARY_PATH:-}"; echo "[slurm] LD_LIBRARY_PATH additions: $LDP"'

ICD_JSON=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
NVIDIA_GLX_LIB=/usr/lib64/libGLX_nvidia.so.0
if [ ! -f "$ICD_JSON" ] || [ ! -f "$NVIDIA_GLX_LIB" ]; then
    echo "ERROR: expected Vulkan ICD file(s) not found on this node ($ICD_JSON, $NVIDIA_GLX_LIB)."
    echo "This diagnostic path is specific to Bridges2 H100 nodes seen so far -- if this fires,"
    echo "the exact host path may differ on this node and needs re-diagnosing (see task-1-report.md)."
    exit 1
fi

RESULTS_DIR="$EVAL_DIR/results/pi05_b1k_smoke/$TASK_NAME"

apptainer exec --nv \
    --bind /ocean/projects/cis250185p/asingal:/ocean/projects/cis250185p/asingal \
    --bind /jet/home/asingal:/jet/home/asingal \
    --bind "$ICD_JSON:$ICD_JSON:ro" \
    --bind "$NVIDIA_GLX_LIB:$NVIDIA_GLX_LIB:ro" \
    --env OMNIGIBSON_HEADLESS=1 \
    --env VK_ICD_FILENAMES="$ICD_JSON" \
    --env CUDA_VISIBLE_DEVICES=1 \
    "$SIF_PATH" \
    bash -c "$LD_LIBRARY_PATH_ADDITIONS_CMD && source /opt/miniconda3/etc/profile.d/conda.sh && conda activate oopsieverse_b1k && python -u '$EVAL_DIR/eval/run_b1k_pi05_eval.py' --tasks $TASK_NAME --num_trials 3 --host localhost --port $PORT --results_dir '$RESULTS_DIR' $DIAGNOSE_FLAG --seed 7"

EVAL_EXIT=$?
echo "=== B1K x pi0.5 smoke test done for $TASK_NAME, exit=$EVAL_EXIT ==="
echo "Check results: $RESULTS_DIR/"
echo "Server log: $SERVER_LOG"
exit $EVAL_EXIT
