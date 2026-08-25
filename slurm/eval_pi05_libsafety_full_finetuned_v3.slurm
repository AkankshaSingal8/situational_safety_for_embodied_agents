#!/bin/bash
# v3: per-EPISODE subprocess isolation, for suites where v2's per-task
# (but multi-episode-per-process) retry still 100%-crashed. Root-caused via
# faulthandler-free but fully reproducible evidence (slurm_logs/
# full_pi05_libsafety_ft_v2_42433116.err): every attempt succeeds on
# episode 0 (video saves, "Episode 0: success=..." logs cleanly), then
# SIGABRTs immediately on transitioning to episode 1 within the SAME env
# object -- 100% reproducible at that exact point, confirmed on both
# obstacle_avoidance (job 42433116) and affordance (job 42433118), across
# two separate full-suite attempts and different nodes (w006, w010). A
# fresh env per episode (fresh subprocess via --episode_offset, added to
# run_libsafety_eval_openpi.py) avoids the reset-reuse path entirely.
#SBATCH --job-name=full_pi05_libsafety_ft_v3
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
NUM_EPISODES=3

echo "=== [job ${SLURM_JOB_ID}] checkpoint=pi05_libero_safety suite=${SUITE} (per-episode retry, v3) ==="

source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh

conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

SERVER_LOG=/tmp/libsafety_full_ft_v3_server_${SLURM_JOB_ID}.log
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
    # Already fully done by an earlier v2 attempt (episode_offset=0 file, no ep-tag)?
    # BUG FIX: the old check here only tested file EXISTENCE, not content --
    # since episode_offset=0 calls reuse the legacy (no-suffix) filename, a
    # task where only episode 0 ever completed left a 1-episode "legacy"
    # file that was then misread as "all 3 episodes done", permanently
    # skipping episodes 1/2 (confirmed: affordance tasks 10/11/13 each only
    # had episode 0 in their legacy file). Now content-aware: count actual
    # episode indices present across BOTH the legacy file and any _epN
    # files for this task, per episode.
    LEGACY_FILE="${LIBSAFETY_DIR}/results/${SUITE}_task${TASK_IDX}_pi05_libero_safety.json"
    for EP in $(seq 0 $((NUM_EPISODES - 1))); do
        EP_FILE="${LIBSAFETY_DIR}/results/${SUITE}_task${TASK_IDX}_ep${EP}_pi05_libero_safety.json"
        ALREADY_DONE=$(python3 -c "
import json
done = False
for fn in ['$LEGACY_FILE', '$EP_FILE']:
    try:
        for r in json.load(open(fn)):
            if r.get('episode') == $EP:
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
            echo "GAVE UP: suite=$SUITE task=$TASK_IDX episode=$EP after $MAX_RETRIES attempts" | tee -a "${LIBSAFETY_DIR}/results/failed_tasks_pi05_ft_v3.log"
        fi
    done
done

echo "=== Suite $SUITE complete, aggregating per-task/episode files ==="
python3 - "$SUITE" "${LIBSAFETY_DIR}/results" << 'PYEOF'
import json, glob, sys
suite, results_dir = sys.argv[1], sys.argv[2]
merged = []
seen = set()
for fn in sorted(glob.glob(f"{results_dir}/{suite}_task*_pi05_libero_safety.json")):
    if "_all_tasks_" in fn:
        continue
    for rec in json.load(open(fn)):
        key = (rec["task_index"], rec["episode"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(rec)
out_fn = f"{results_dir}/{suite}_all_tasks_pi05_libero_safety.json"
json.dump(merged, open(out_fn, "w"), indent=2)
print(f"Wrote {len(merged)} episodes to {out_fn}")
PYEOF
