#!/bin/bash
# Smoke test: pi0.5 fine-tuned on LIBERO-Safety's own training data
# (LIBERO-Safety/pi05_libero_safety, HF), on ONE task of ONE suite, to
# confirm the checkpoint loads and produces non-degenerate (non-zero)
# results before committing to the full 4-suite sweep.
#
# Architecture matches the off-the-shelf pi0.5 runs (task9b_full_libsafety_*
# jobs): openpi policy server (libsafety_openpi env) + eval client
# (libsafety_client env) talking over websocket on localhost.
#
# --policy.config stays "pi05_libero" (selects the pi0.5-LIBERO model
# architecture/input transforms) while --policy.dir points at the
# LIBERO-Safety fine-tuned WEIGHTS -- config and checkpoint dir are
# decoupled here (unlike libsafety_serve_openpi.sh, which assumes they
# share a name) because the fine-tuned checkpoint keeps the same
# architecture, just different weights.
#SBATCH --job-name=smoke_pi05_libsafety_ft
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=00:30:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.err

set -uo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE=$REPO/.worktrees/libero-safety-benchmark
LIBSAFETY_DIR=$WORKTREE/LIBERO-Safety
PORT=8000
SUITE=human_safety

echo "=== [job ${SLURM_JOB_ID}] smoke test: pi05_libero_safety (fine-tuned), suite=${SUITE} task=0 ==="

source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh

# ── Step 1: server (libsafety_openpi env) ─────────────────────────────────
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

SERVER_LOG=/tmp/libsafety_ft_smoke_server_${SLURM_JOB_ID}.log
python "$REPO/vlsa-aegis/openpi/scripts/serve_policy.py" \
    --port="${PORT}" \
    policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir="${LIBSAFETY_DIR}/checkpoints/pi05_libero_safety" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "=== Starting server (pi05_libero_safety) in background, log=$SERVER_LOG, pid=$SERVER_PID ==="

cleanup() {
    echo "=== Stopping server (pid=$SERVER_PID) ==="
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Waiting for server to report listening ==="
MAX_WAIT=600
ELAPSED=0
until grep -qi "server listening on" "$SERVER_LOG" 2>/dev/null || ! kill -0 "$SERVER_PID" 2>/dev/null; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: server did not report listening within ${MAX_WAIT}s"
        cat "$SERVER_LOG"
        exit 1
    fi
done
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: server process died during startup"
    cat "$SERVER_LOG"
    exit 1
fi
echo "=== Server reported listening after ${ELAPSED}s ==="

# ── Step 2: client (libsafety_client env), single task ────────────────────
conda deactivate
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_client
export MUJOCO_GL=egl
export LD_PRELOAD=/ocean/projects/cis250185p/asingal/envs/libsafety_client/lib/libstdc++.so.6
export LIBERO_CONFIG_PATH="${LIBSAFETY_DIR}/.libero_config"

cd "$WORKTREE"
echo "=== Running client (task 0 only, 1 trial) ==="
python run_libsafety_eval_openpi.py \
    --task_suite_name "$SUITE" \
    --task_index 0 \
    --num_trials_per_task 1 \
    --host localhost \
    --port "$PORT" \
    --checkpoint_name pi05_libero_safety \
    --results_out_path "${LIBSAFETY_DIR}/results" \
    --video_out_path "${LIBSAFETY_DIR}/rollouts" \
    --seed 7
CLIENT_EXIT=$?
echo "=== Client exited with code $CLIENT_EXIT ==="

exit $CLIENT_EXIT
