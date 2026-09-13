#!/bin/bash
# Full AEGIS (ground-truth-obstacle variant) eval on LIBERO-Safety, 4 suites
# x 15 tasks x 3 trials, off-the-shelf pi05_libero checkpoint. Per-task
# retry architecture from the start (same native EGL/render crash class
# already seen twice -- Fast-WAM, pi0.5 fine-tuned -- so we don't burn a
# whole-suite job discovering it a third time).
#SBATCH --job-name=full_aegis_libsafety
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=08:00:00
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

echo "=== [job ${SLURM_JOB_ID}] AEGIS (ground-truth obstacle) checkpoint=pi05_libero suite=${SUITE} trials=3 (per-task retry) ==="

source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh

conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

SERVER_LOG=/tmp/libsafety_aegis_full_server_${SLURM_JOB_ID}.log
python "$REPO/vlsa-aegis/openpi/scripts/serve_policy.py" \
    --port="${PORT}" \
    policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir="${LIBSAFETY_DIR}/checkpoints/pi05_libero" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "=== Starting server (pi05_libero, off-the-shelf) in background, log=$SERVER_LOG, pid=$SERVER_PID ==="

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

conda deactivate
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_client
export MUJOCO_GL=egl
export LD_PRELOAD=/ocean/projects/cis250185p/asingal/envs/libsafety_client/lib/libstdc++.so.6
export LIBERO_CONFIG_PATH="${LIBSAFETY_DIR}/.libero_config"

cd "$WORKTREE"

for TASK_IDX in $(seq 0 $((NUM_TASKS - 1))); do
    SUCCESS=0
    for ATTEMPT in $(seq 1 $MAX_RETRIES); do
        echo "=== Suite $SUITE  Task $TASK_IDX  Attempt $ATTEMPT/$MAX_RETRIES ==="
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server died -- cannot continue this suite"
            cat "$SERVER_LOG"
            exit 1
        fi
        python run_libsafety_eval_aegis_gt.py \
            --task_suite_name "$SUITE" \
            --task_index "$TASK_IDX" \
            --num_trials_per_task 3 \
            --host localhost \
            --port "$PORT" \
            --checkpoint_name aegis_gt_pi05_libero \
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
        echo "GAVE UP: suite=$SUITE task=$TASK_IDX after $MAX_RETRIES attempts" | tee -a "${LIBSAFETY_DIR}/results/failed_tasks_aegis.log"
    fi
done

echo "=== Suite $SUITE complete, aggregating per-task files ==="
python3 - "$SUITE" "${LIBSAFETY_DIR}/results" << 'PYEOF'
import json, glob, sys
suite, results_dir = sys.argv[1], sys.argv[2]
merged = []
for fn in sorted(glob.glob(f"{results_dir}/{suite}_task*_aegis_gt_pi05_libero.json")):
    merged.extend(json.load(open(fn)))
out_fn = f"{results_dir}/{suite}_all_tasks_aegis_gt_pi05_libero.json"
json.dump(merged, open(out_fn, "w"), indent=2)
print(f"Wrote {len(merged)} episodes to {out_fn}")
PYEOF
