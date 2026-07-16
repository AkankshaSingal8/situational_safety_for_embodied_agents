#!/bin/bash
# Parameterized Pi0.5 evaluation on SafeLIBERO.
# Starts the Pi0.5 WebSocket server (aegis env), waits for it to be ready,
# then runs the eval client (libero env), and kills the server on exit.
#
# Called by submit_all_evals.sh with --export=ALL,SUITE=...,LEVEL=...
# GPU: h100-80 — Pi0.5 requires large VRAM.
#SBATCH --job-name=pi05_eval
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=12:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

# ── Validate parameters ────────────────────────────────────────────────────────
: "${SUITE:?ERROR: SUITE env var not set.}"
: "${LEVEL:?ERROR: LEVEL env var not set.}"

echo "=== Pi0.5 Eval: SUITE=$SUITE, LEVEL=$LEVEL ==="

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
AEGIS=$REPO/vlsa-aegis
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/pi05_server_${SLURM_JOB_ID}.log

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p "$REPO/pi05_benchmark"
mkdir -p "$REPO/pi05_video"

module load anaconda3

# ── Step 1: Start Pi0.5 server (aegis env) ────────────────────────────────────
conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONPATH="$AEGIS/openpi/src:${PYTHONPATH:-}"

# Kill any stale process on port 8000 left by a previous job on this node.
# Without this, the new server gets EADDRINUSE but nc -z still sees the old
# server — client connects to the zombie, which dies mid-run and corrupts results.
fuser -k 8000/tcp 2>/dev/null || true
sleep 2

echo "Starting Pi0.5 server..."
python "$AEGIS/openpi/scripts/serve_policy.py" \
    policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir="$AEGIS/checkpoints/pi05_libero" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID  (log: $SERVER_LOG)"

cleanup() {
    echo "Stopping Pi0.5 server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Step 2: Wait for server to be ready (max 10 min) ──────────────────────────
echo "Waiting for Pi0.5 server on port 8000..."
MAX_WAIT=600
ELAPSED=0
until nc -z localhost 8000 2>/dev/null; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  ...${ELAPSED}s"
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Server did not start within ${MAX_WAIT}s. Server log:"
        cat "$SERVER_LOG"
        exit 1
    fi
done
echo "Pi0.5 server is ready!"

# ── Step 3: Run eval client (libero env) ─────────────────────────────────────
# LD_PRELOAD: system /lib64/libstdc++.so.6 lacks GLIBCXX_3.4.29 (required by
# PIL/libLerc); preload the conda env's newer libstdc++ to fix the ImportError.
conda activate libero
export MUJOCO_GL=egl
export LD_PRELOAD=/ocean/projects/cis250185p/asingal/envs/libero/lib/libstdc++.so.6
export PYTHONPATH="$AEGIS/openpi/src:$REPO/SafeLIBERO/safelibero:$REPO/vlm_pipeline:${PYTHONPATH:-}"

python "$REPO/vlm_pipeline/run_safelibero_pi05_eval.py" \
    --task_suite_name "$SUITE" \
    --safety_level "$LEVEL" \
    --num_trials_per_task 50 \
    --host localhost \
    --port 8000 \
    --results_output_dir "$REPO/pi05_benchmark" \
    --video_output_dir "$REPO/pi05_video" \
    --seed 7

EVAL_EXIT=$?
echo "=== Done: SUITE=$SUITE, LEVEL=$LEVEL, exit=$EVAL_EXIT ==="
exit $EVAL_EXIT
