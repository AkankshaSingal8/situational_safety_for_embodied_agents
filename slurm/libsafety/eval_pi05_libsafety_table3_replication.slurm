#!/bin/bash
# Table-3 replication run: pi0.5 fine-tuned (pi05_libero_safety checkpoint) on
# LIBERO-Safety, 10 trials/task (matching the paper's protocol -- see
# arXiv:2606.23686 Sec 4.1: "Each task is evaluated over 10 independent
# trials"), with collision/cost tracking added to run_libsafety_eval_openpi.py
# so we can compute the paper's SR = success AND NOT collision, not just TSR.
#
# Uses a SEPARATE checkpoint_name ("pi05_libero_safety_10trial") from the
# earlier 3-trial task#22 run ("pi05_libero_safety") so this does not clobber
# or get confused with that data -- the old 3-trial files lack the
# `collision` field entirely and are kept as-is for the other comparison
# table.
#
# Per-EPISODE subprocess isolation (v3 pattern): reusing one env across
# episodes SIGABRTs reliably on episode 2 (root-caused earlier this session).
#SBATCH --job-name=pi05_libsafety_table3
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
NUM_EPISODES=10
CKPT_NAME=pi05_libero_safety_10trial

echo "=== [job ${SLURM_JOB_ID}] Table-3 replication checkpoint=pi05_libero_safety suite=${SUITE} (10 trials/task, per-episode retry) ==="

source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh

conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

SERVER_LOG=/tmp/pi05_libsafety_table3_server_${SLURM_JOB_ID}.log
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

conda deactivate
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_client
export MUJOCO_GL=egl
export LD_PRELOAD=/ocean/projects/cis250185p/asingal/envs/libsafety_client/lib/libstdc++.so.6
export LIBERO_CONFIG_PATH="${LIBSAFETY_DIR}/.libero_config"

cd "$WORKTREE"

for TASK_IDX in $(seq 0 $((NUM_TASKS - 1))); do
    LEGACY_FILE="${LIBSAFETY_DIR}/results/${SUITE}_task${TASK_IDX}_${CKPT_NAME}.json"
    for EP in $(seq 0 $((NUM_EPISODES - 1))); do
        EP_FILE="${LIBSAFETY_DIR}/results/${SUITE}_task${TASK_IDX}_ep${EP}_${CKPT_NAME}.json"
        ALREADY_DONE=$(python3 -c "
import json
done = False
for fn in ['$LEGACY_FILE', '$EP_FILE']:
    try:
        for r in json.load(open(fn)):
            if r.get('episode') == $EP and 'collision' in r:
                done = True
    except FileNotFoundError:
        pass
print('yes' if done else 'no')
")
        if [ "$ALREADY_DONE" = "yes" ]; then
            echo "=== Suite $SUITE  Task $TASK_IDX  Episode $EP  already done, skipping ==="
            continue
        fi
        SUCCESS=0
        for ATTEMPT in $(seq 1 $MAX_RETRIES); do
            echo "=== Suite $SUITE  Task $TASK_IDX  Episode $EP  Attempt $ATTEMPT/$MAX_RETRIES ==="
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                echo "ERROR: server died -- cannot continue this suite"
                cat "$SERVER_LOG"
                exit 1
            fi
            python run_libsafety_eval_openpi.py \
                --task_suite_name "$SUITE" \
                --task_index "$TASK_IDX" \
                --num_trials_per_task 1 \
                --episode_offset "$EP" \
                --host localhost \
                --port "$PORT" \
                --checkpoint_name "$CKPT_NAME" \
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
            echo "GAVE UP: suite=$SUITE task=$TASK_IDX episode=$EP after $MAX_RETRIES attempts" | tee -a "${LIBSAFETY_DIR}/results/failed_tasks_pi05_table3.log"
        fi
    done
done

echo "=== Suite $SUITE complete, aggregating per-task/episode files ==="
python3 - "$SUITE" "${LIBSAFETY_DIR}/results" "$CKPT_NAME" << 'PYEOF'
import json, glob, sys
suite, results_dir, ckpt_name = sys.argv[1], sys.argv[2], sys.argv[3]
merged = []
seen = set()
for fn in sorted(glob.glob(f"{results_dir}/{suite}_task*_{ckpt_name}.json")):
    if "_all_tasks_" in fn:
        continue
    for rec in json.load(open(fn)):
        key = (rec["task_index"], rec["episode"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(rec)
out_fn = f"{results_dir}/{suite}_all_tasks_{ckpt_name}.json"
json.dump(merged, open(out_fn, "w"), indent=2)
print(f"Wrote {len(merged)} episodes to {out_fn}")
PYEOF
