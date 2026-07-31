#!/bin/bash
# Full pi0.5 RoboCasa evaluation: all 15 tasks (shelve_item excluded) x 30
# trials = 450 episodes.
#
# SLURM ARRAY (one array index per task), following eval_full_openvla_robocasa
# .slurm's pattern: 30 trials/task is 3x this repo's earlier 10-trial full
# runs, so even though pi0.5 is fast per-episode, splitting into 15
# independent ~1hr shards still lets them run concurrently (subject to
# cluster availability) instead of serially, and a crash in one task's shard
# doesn't lose the other 14. Each shard writes to its own results subdir
# (see below) -- critical: without this, 15 shards racing on one
# episodes.json would silently keep only 1/15 of the results.
#SBATCH --job-name=oopsieverse_pi05_fullsuite
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=02:00:00
#SBATCH --array=0-14
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval
EVAL_DIR="$REPO/oopsieverse_eval"
# Distinct port band from pi05 paper-match (8201) and any other array job.
PORT=$((8500 + SLURM_ARRAY_TASK_ID))
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/pi05_server_fullsuite_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

# shelve_item excluded -- needs a camera-calibration file OopsieVerse doesn't
# ship. Must exactly match ALL_TASKS in run_robocasa_pi05_eval.py.
TASKS=(pick_egg serve_pastry open_single_door turn_on_faucet turn_on_microwave turn_on_stove
       open_drawer close_drawer place_plate counter_to_microwave prepare_coffee
       prepare_breakfast dishes_to_sink nav_lift_bowl wipe_counter)
TASK_NAME="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== pi0.5 full-suite eval: task=$TASK_NAME (array idx $SLURM_ARRAY_TASK_ID), port=$PORT ==="

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

module load anaconda3

conda activate /ocean/projects/cis250185p/asingal/envs/oopsieverse_pi05_srv
export OPENPI_DATA_HOME=/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

bash "$EVAL_DIR/servers/launch_pi05_server.sh" "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for server on port $PORT..."
MAX_WAIT=1800
ELAPSED=0
until nc -z localhost "$PORT" 2>/dev/null; do
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Server did not start. Log:"; cat "$SERVER_LOG"; exit 1
    fi
done
echo "Server ready!"

conda activate /ocean/projects/cis250185p/asingal/envs/oopsieverse_robocasa
export MUJOCO_GL=egl

python "$EVAL_DIR/eval/run_robocasa_pi05_eval.py" \
    --tasks "$TASK_NAME" \
    --num_trials 30 \
    --host localhost --port "$PORT" \
    --results_dir "$EVAL_DIR/results/pi05_robocasa_fullsuite/$TASK_NAME" \
    --seed 7

EXIT=$?
echo "=== $TASK_NAME done, exit=$EXIT ==="
exit $EXIT
