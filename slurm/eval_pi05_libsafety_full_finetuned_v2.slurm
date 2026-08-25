#!/bin/bash
# v2: crash-resilient version of eval_pi05_libsafety_full_finetuned.slurm.
# All 4 first-attempt jobs (42431576/90/91/92) died with SIGABRT (exit 134)
# partway through the --all_tasks sweep -- same native EGL/rendering crash
# class as Fast-WAM's (see eval_fastwam_libsafety_full_v2.slurm). Episode 0
# of task 0 always completed and saved fine before the abort, confirming
# the SERVER survives; only the CLIENT process dies.
#
# Fix: keep ONE server alive per suite (as before), but loop task_index
# 0..14 with a fresh client process + retry-on-abort per task, instead of
# a single --all_tasks client process for the whole suite. A crash now
# only loses one task's in-flight trial, not the whole suite.
#SBATCH --job-name=full_pi05_libsafety_ft_v2
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=12:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.err

set -uo pipefail

: "${SUITE:?ERROR: SUITE env var not set (export via --export=ALL,SUITE=...)}"

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE=$REPO/.worktrees/libero-safety-benchmark
LIBSAFETY_DIR=$WORKTREE/LIBERO-Safety
PORT="${PORT:-$((8000 + SLURM_JOB_ID % 1000))}"
MAX_RETRIES=3
NUM_TASKS=15

echo "=== [job ${SLURM_JOB_ID}] checkpoint=pi05_libero_safety suite=${SUITE} trials=3 (per-task retry) ==="

source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh

# ── Step 1: server (libsafety_openpi env), stays up for the whole suite ───
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

SERVER_LOG=/tmp/libsafety_full_ft_v2_server_${SLURM_JOB_ID}.log
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

# ── Step 2: client (libsafety_client env), one task at a time + retry ─────
conda deactivate
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_client
export MUJOCO_GL=egl
export LD_PRELOAD=/ocean/projects/cis250185p/asingal/envs/libsafety_client/lib/libstdc++.so.6
export LIBERO_CONFIG_PATH="${LIBSAFETY_DIR}/.libero_config"

cd "$WORKTREE"

for TASK_IDX in $(seq 0 $((NUM_TASKS - 1))); do
    EXISTING="${LIBSAFETY_DIR}/results/${SUITE}_task${TASK_IDX}_pi05_libero_safety.json"
    if [ -f "$EXISTING" ]; then
        echo "=== Suite $SUITE  Task $TASK_IDX  already done (resume), skipping ==="
        continue
    fi
    SUCCESS=0
    for ATTEMPT in $(seq 1 $MAX_RETRIES); do
        echo "=== Suite $SUITE  Task $TASK_IDX  Attempt $ATTEMPT/$MAX_RETRIES ==="
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server died -- cannot continue this suite"
            cat "$SERVER_LOG"
            exit 1
        fi
        python run_libsafety_eval_openpi.py \
            --task_suite_name "$SUITE" \
            --task_index "$TASK_IDX" \
            --num_trials_per_task 3 \
            --host localhost \
            --port "$PORT" \
            --checkpoint_name pi05_libero_safety \
            --results_out_path "${LIBSAFETY_DIR}/results" \
            --video_out_path "${LIBSAFETY_DIR}/rollouts" \
            --seed 7
        EXIT=$?
        if [ $EXIT -eq 0 ]; then
            SUCCESS=1
            break
        else
            echo "Attempt $ATTEMPT failed (exit=$EXIT), retrying in 10s..."
            sleep 10
        fi
    done
    if [ $SUCCESS -eq 0 ]; then
        echo "GAVE UP: suite=$SUITE task=$TASK_IDX after $MAX_RETRIES attempts" | tee -a "${LIBSAFETY_DIR}/results/failed_tasks_pi05_ft.log"
    fi
done

echo "=== Suite $SUITE complete, aggregating per-task files ==="
python3 - "$SUITE" "${LIBSAFETY_DIR}/results" << 'PYEOF'
import json, glob, sys
suite, results_dir = sys.argv[1], sys.argv[2]
merged = []
for fn in sorted(glob.glob(f"{results_dir}/{suite}_task*_pi05_libero_safety.json")):
    merged.extend(json.load(open(fn)))
out_fn = f"{results_dir}/{suite}_all_tasks_pi05_libero_safety.json"
json.dump(merged, open(out_fn, "w"), indent=2)
print(f"Wrote {len(merged)} episodes to {out_fn}")
PYEOF
