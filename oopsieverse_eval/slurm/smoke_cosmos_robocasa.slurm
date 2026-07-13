#!/bin/bash
# Cosmos-Policy-RoboCasa (off-the-shelf, nvidia/Cosmos-Policy-RoboCasa-Predict2-2B)
# smoke test on OopsieVerse RoboCasa: 3 episodes each on 2 representative
# tasks. Server runs inside cosmos-policy's existing, already-debugged
# Apptainer container/.venv_rhel8 (read-only reuse, no rebuild); client runs
# in the already-built oopsieverse_robocasa env. NOTE: this is an
# out-of-task-distribution baseline (see run_robocasa_cosmos_eval.py's
# docstring) -- report accordingly, don't oversell as in-distribution.
#SBATCH --job-name=oopsieverse_cosmos_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=63000M
#SBATCH --time=01:30:00
# NOTE: an "Aborted (core dumped)" (SIGABRT, no Python traceback) has now
# been seen on w006 (twice: pi0.5 full-eval, an earlier attempt of this job)
# AND w008 (this job's most recent attempt) -- so it is NOT a single bad
# node; --exclude=w006 below is left in only because it's free, not because
# it's expected to fix anything. The w008 crash happened deep into the run
# (server-side log shows 51 successful inference calls before it, ruling out
# a startup-time issue) and Cosmos's RoboCasa client is the only one of the
# four policies using 3 cameras (agentview_left + agentview_right +
# eye_in_hand) instead of 2 -- suspect MuJoCo/EGL offscreen-render resource
# exhaustion under sustained use, not yet confirmed. See PYTHONUNBUFFERED
# and ulimit -c below: added to get a real crash location/core dump on the
# next occurrence instead of guessing further.
#SBATCH --exclude=w006
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail
echo "=== OopsieVerse RoboCasa x Cosmos-Policy smoke test ==="

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
EVAL_DIR="$REPO/.worktrees/oopsieverse-eval/oopsieverse_eval"
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/cosmos_server_${SLURM_JOB_ID}.log
PORT=8400

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

module load anaconda3

# ── Start Cosmos server (inside Apptainer, read-only reuse) ─────────────
bash "$EVAL_DIR/servers/launch_cosmos_server.sh" "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

cleanup() {
    echo "Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server on port $PORT..."
MAX_WAIT=1200
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
# Unbuffered: the last "Aborted (core dumped)" run showed ZERO episode
# markers in this log despite the server having handled 51 successful
# inference calls -- Python fully block-buffers stdout when it's not a
# TTY (i.e. always, when redirected to a SLURM log file), so a native
# SIGABRT can wipe out unflushed print() output and make the crash look
# like it happened immediately when it actually happened much later.
export PYTHONUNBUFFERED=1
# Enable core dumps so a repeat crash leaves something to inspect (gdb
# batch-mode backtrace, python-side frame info via `py-bt` if debug symbols
# are available) instead of just "Aborted (core dumped)" with no detail.
ulimit -c unlimited
CORE_DIR="/ocean/projects/cis250185p/asingal/slurm_logs/cores_${SLURM_JOB_ID}"
mkdir -p "$CORE_DIR"
cd "$CORE_DIR"

python -u "$EVAL_DIR/eval/run_robocasa_cosmos_eval.py" \
    --tasks pick_egg turn_on_faucet \
    --num_trials 3 \
    --host localhost --port "$PORT" \
    --results_dir "$EVAL_DIR/results/cosmos_robocasa_smoke" \
    --seed 7

EVAL_EXIT=$?
echo "=== Smoke test done, exit=$EVAL_EXIT ==="
echo "Check results: $EVAL_DIR/results/cosmos_robocasa_smoke/"
echo "Server log: $SERVER_LOG"
exit $EVAL_EXIT
