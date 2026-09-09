#!/bin/bash
# Full pi0.5-LIBERO-Safety fine-tuned eval (LIBERO-Safety/pi05_libero_safety),
# 4 physical-safety suites x all_tasks sweep x 3 trials/task, matching the
# off-the-shelf pi05_libero protocol already in the results table.
# reasoning_safety (SSR) excluded: different eval paradigm (refusal-rate
# VQA judgment, not rollout success), unsupported by this rollout client.
#
# One SLURM job per suite (submit_all via a wrapper loop), server+client
# pair, same architecture as the smoke test that just passed (job 42431534).
#SBATCH --job-name=full_pi05_libsafety_ft
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.err

set -uo pipefail

: "${SUITE:?ERROR: SUITE env var not set (export via --export=ALL,SUITE=...)}"

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE=$REPO/.worktrees/libero-safety-benchmark
LIBSAFETY_DIR=$WORKTREE/LIBERO-Safety
# Multiple suite jobs can land on the same shared GPU node (confirmed:
# jobs 42431574-77 all landed on w006), so a hardcoded port causes 3/4 to
# lose the bind race. Default to a per-job unique port; override via
# --export=ALL,SUITE=...,PORT=... if a specific port is needed.
PORT="${PORT:-$((8000 + SLURM_JOB_ID % 1000))}"

echo "=== [job ${SLURM_JOB_ID}] checkpoint=pi05_libero_safety suite=${SUITE} trials=3 (all_tasks) ==="

source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh

# ── Step 1: server (libsafety_openpi env) ─────────────────────────────────
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

SERVER_LOG=/tmp/libsafety_full_ft_server_${SLURM_JOB_ID}.log
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

# ── Step 2: client (libsafety_client env), all_tasks sweep ────────────────
conda deactivate
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_client
export MUJOCO_GL=egl
export LD_PRELOAD=/ocean/projects/cis250185p/asingal/envs/libsafety_client/lib/libstdc++.so.6
export LIBERO_CONFIG_PATH="${LIBSAFETY_DIR}/.libero_config"

cd "$WORKTREE"
echo "=== Running client (all_tasks sweep) ==="
python run_libsafety_eval_openpi.py \
    --task_suite_name "$SUITE" \
    --all_tasks \
    --num_trials_per_task 3 \
    --host localhost \
    --port "$PORT" \
    --checkpoint_name pi05_libero_safety \
    --results_out_path "${LIBSAFETY_DIR}/results" \
    --video_out_path "${LIBSAFETY_DIR}/rollouts" \
    --seed 7
CLIENT_EXIT=$?
echo "=== Client exited with code $CLIENT_EXIT ==="

exit $CLIENT_EXIT
