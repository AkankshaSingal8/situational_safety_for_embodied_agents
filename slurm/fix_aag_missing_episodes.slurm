#!/bin/bash
# Targeted completion of the 4 specific (task,episode) pairs missing from
# affordance due to the legacy-filename collision bug in v3's resume
# check (existence-only, not content-aware -- task10/11/13's "legacy"
# files only ever got episode 0 written, then were misread as "task done").
#SBATCH --job-name=fix_aag_missing
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=02:00:00
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

SERVER_LOG=/tmp/fix_aag_server_${SLURM_JOB_ID}.log
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

# (task, episode) pairs -- fixed list, this is a one-off cleanup, not a generic loop.
PAIRS="10:2 11:2 13:1 13:2"
for PAIR in $PAIRS; do
    T="${PAIR%%:*}"
    E="${PAIR##*:}"
    for ATTEMPT in 1 2 3; do
        echo "=== task=$T episode=$E attempt=$ATTEMPT ==="
        python run_libsafety_eval_openpi.py \
            --task_suite_name affordance --task_index "$T" \
            --num_trials_per_task 1 --episode_offset "$E" \
            --host localhost --port "$PORT" \
            --checkpoint_name pi05_libero_safety_fixep \
            --results_out_path "${LIBSAFETY_DIR}/results" \
            --video_out_path "${LIBSAFETY_DIR}/rollouts" --seed 7
        [ $? -eq 0 ] && break
        sleep 10
    done
done

echo "=== Merging fixed episodes into the main affordance results file ==="
python3 << 'PYEOF'
import json, glob
results_dir = "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/LIBERO-Safety/results"
main_fn = f"{results_dir}/affordance_all_tasks_pi05_libero_safety.json"
merged = json.load(open(main_fn))
seen = {(r["task_index"], r["episode"]) for r in merged}
for fn in glob.glob(f"{results_dir}/affordance_task*_pi05_libero_safety_fixep.json"):
    for rec in json.load(open(fn)):
        key = (rec["task_index"], rec["episode"])
        if key not in seen:
            rec["checkpoint"] = "pi05_libero_safety"
            merged.append(rec)
            seen.add(key)
json.dump(merged, open(main_fn, "w"), indent=2)
print(f"Final: {len(merged)} episodes in {main_fn}")
PYEOF
