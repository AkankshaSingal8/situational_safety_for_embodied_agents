#!/bin/bash
# OpenVLA (off-the-shelf, openvla-7b-finetuned-libero-10) smoke test on
# OopsieVerse RoboCasa: 3 episodes each on 2 representative tasks, to
# validate the server + client + adapter wiring before scaling to the full
# 16-task suite. Uses two fresh, isolated envs (oopsieverse_openvla_srv for
# the server, oopsieverse_robocasa for the sim client) -- neither touches
# any existing env in this repo.
#SBATCH --job-name=oopsieverse_openvla_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail
echo "=== OopsieVerse RoboCasa x OpenVLA smoke test ==="

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval
EVAL_DIR="$REPO/oopsieverse_eval"
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/openvla_server_${SLURM_JOB_ID}.log
PORT=8100

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

module load anaconda3

# ── Start OpenVLA REST server ────────────────────────────────────────────
conda activate /ocean/projects/cis250185p/asingal/envs/oopsieverse_openvla_srv
export HF_HOME=/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/hf
export TRANSFORMERS_CACHE="$HF_HOME"

echo "Starting OpenVLA server..."
python "$EVAL_DIR/servers/openvla_server.py" \
    --checkpoint openvla/openvla-7b-finetuned-libero-10 \
    --unnorm_key libero_10 \
    --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

cleanup() {
    echo "Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server on port $PORT..."
MAX_WAIT=900
ELAPSED=0
until curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; do
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

python "$EVAL_DIR/eval/run_robocasa_openvla_eval.py" \
    --tasks pick_egg turn_on_faucet \
    --num_trials 3 \
    --server_url "http://localhost:${PORT}" \
    --results_dir "$EVAL_DIR/results/openvla_robocasa_smoke" \
    --seed 7

EVAL_EXIT=$?
echo "=== Smoke test done, exit=$EVAL_EXIT ==="
echo "Check results: $EVAL_DIR/results/openvla_robocasa_smoke/"
echo "Server log: $SERVER_LOG"
exit $EVAL_EXIT
