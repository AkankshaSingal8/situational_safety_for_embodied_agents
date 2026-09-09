#!/bin/bash
# Smoke test: confirm the new `collision` field in run_libsafety_eval_openpi.py
# actually populates (not just always False), before submitting the full
# 600-episode Table-3 replication run. 1 task, 2 episodes.
#SBATCH --job-name=smoke_pi05_table3
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
PORT="${PORT:-$((8000 + SLURM_JOB_ID % 1000))}"

source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

SERVER_LOG=/tmp/smoke_pi05_table3_server_${SLURM_JOB_ID}.log
python "$REPO/vlsa-aegis/openpi/scripts/serve_policy.py" \
    --port="${PORT}" policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir="${LIBSAFETY_DIR}/checkpoints/pi05_libero_safety" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

MAX_WAIT=600; ELAPSED=0
until grep -qi "server listening on" "$SERVER_LOG" 2>/dev/null || ! kill -0 "$SERVER_PID" 2>/dev/null; do
    sleep 5; ELAPSED=$((ELAPSED + 5))
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then cat "$SERVER_LOG"; exit 1; fi
done
echo "=== Server ready after ${ELAPSED}s ==="

conda deactivate
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_client
export MUJOCO_GL=egl
export LD_PRELOAD=/ocean/projects/cis250185p/asingal/envs/libsafety_client/lib/libstdc++.so.6
export LIBERO_CONFIG_PATH="${LIBSAFETY_DIR}/.libero_config"
cd "$WORKTREE"

for EP in 0 1; do
    echo "=== smoke episode $EP ==="
    python run_libsafety_eval_openpi.py \
        --task_suite_name obstacle_avoidance --task_index 2 \
        --num_trials_per_task 1 --episode_offset "$EP" \
        --host localhost --port "$PORT" \
        --checkpoint_name pi05_libero_safety_smoke \
        --results_out_path "${LIBSAFETY_DIR}/results" \
        --video_out_path "${LIBSAFETY_DIR}/rollouts" --seed 7
done

echo "=== Smoke results ==="
python3 -c "
import json, glob
for fn in sorted(glob.glob('${LIBSAFETY_DIR}/results/obstacle_avoidance_task2*pi05_libero_safety_smoke.json')):
    for r in json.load(open(fn)):
        print(fn, '-> success=', r['success'], 'collision=', r.get('collision', 'MISSING_FIELD'))
"
