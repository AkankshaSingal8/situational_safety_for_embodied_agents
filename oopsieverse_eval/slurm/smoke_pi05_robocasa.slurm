#!/bin/bash
# pi0.5 (off-the-shelf, pi05_libero) smoke test on OopsieVerse RoboCasa: 3
# episodes each on 2 representative tasks, to validate the server + client
# + adapter wiring before scaling to the full 16-task suite. Uses two
# fresh, isolated envs (oopsieverse_pi05_srv for the server,
# oopsieverse_robocasa for the sim client) -- neither touches any existing
# env in this repo. Uses H100 -- pi0.5 needs large VRAM.
#SBATCH --job-name=oopsieverse_pi05_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=01:30:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail
echo "=== OopsieVerse RoboCasa x pi0.5 smoke test ==="

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval
EVAL_DIR="$REPO/oopsieverse_eval"
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/pi05_server_${SLURM_JOB_ID}.log
PORT=8200

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

module load anaconda3

# ── Start pi0.5 server (openpi's own serve_policy.py, unmodified) ────────
conda activate /ocean/projects/cis250185p/asingal/envs/oopsieverse_pi05_srv
export OPENPI_DATA_HOME=/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "Starting pi0.5 server (downloads pi05_libero from GCS on first run)..."
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

# ── Run eval client (sim-host env) ───────────────────────────────────────
conda activate /ocean/projects/cis250185p/asingal/envs/oopsieverse_robocasa
export MUJOCO_GL=egl

python "$EVAL_DIR/eval/run_robocasa_pi05_eval.py" \
    --tasks pick_egg turn_on_faucet \
    --num_trials 3 \
    --host localhost --port "$PORT" \
    --results_dir "$EVAL_DIR/results/pi05_robocasa_smoke" \
    --seed 7

EVAL_EXIT=$?
echo "=== Smoke test done, exit=$EVAL_EXIT ==="
echo "Check results: $EVAL_DIR/results/pi05_robocasa_smoke/"
echo "Server log: $SERVER_LOG"
exit $EVAL_EXIT
