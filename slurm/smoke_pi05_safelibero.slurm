#!/bin/bash
# Pi0.5 smoke test: 5 episodes on spatial Level I, task 0 only.
# Run this before submitting the full sweep to validate server + eval infrastructure.
# Results go to pi05_benchmark_smoke/ (separate from main benchmark dir).
# Uses H100 — Pi0.5 needs large VRAM.
#SBATCH --job-name=pi05_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=02:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

echo "=== Pi0.5 Smoke Test: spatial Level I, task 0, 5 episodes ==="

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
AEGIS=$REPO/vlsa-aegis
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/pi05_server_${SLURM_JOB_ID}.log

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p "$REPO/pi05_benchmark_smoke"

module load anaconda3

# ── Start Pi0.5 server (aegis env) ────────────────────────────────────────────
conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONPATH="$AEGIS/openpi/src:${PYTHONPATH:-}"

echo "Starting Pi0.5 server..."
python "$AEGIS/openpi/scripts/serve_policy.py" \
    policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir="$AEGIS/checkpoints/pi05_libero" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

cleanup() {
    echo "Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Wait for server ready (max 10 min) ────────────────────────────────────────
echo "Waiting for server on port 8000..."
MAX_WAIT=600
ELAPSED=0
until nc -z localhost 8000 2>/dev/null; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  ...${ELAPSED}s"
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Server did not start. Log:"
        cat "$SERVER_LOG"
        exit 1
    fi
done
echo "Server ready!"

# ── Run eval client (libero env) ─────────────────────────────────────────────
# libero env needs its own libstdc++ — the system's /lib64/libstdc++.so.6
# lacks GLIBCXX_3.4.29 required by PIL/libLerc on this cluster.
conda activate libero
export MUJOCO_GL=egl
export LD_PRELOAD=/ocean/projects/cis250185p/asingal/envs/libero/lib/libstdc++.so.6
export PYTHONPATH="$AEGIS/openpi/src:$REPO/SafeLIBERO/safelibero:$REPO/vlm_pipeline:${PYTHONPATH:-}"

python "$REPO/vlm_pipeline/run_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 5 \
    --task_indices 0 \
    --host localhost \
    --port 8000 \
    --results_output_dir "$REPO/pi05_benchmark_smoke" \
    --seed 7

EVAL_EXIT=$?
echo "=== Smoke test done, exit=$EVAL_EXIT ==="
echo "Check results: $REPO/pi05_benchmark_smoke/safelibero_spatial/I/"
exit $EVAL_EXIT
