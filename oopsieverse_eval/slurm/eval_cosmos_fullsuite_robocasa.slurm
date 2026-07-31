#!/bin/bash
# Full Cosmos-Policy-RoboCasa evaluation: all 15 tasks (shelve_item excluded)
# x 30 trials = 450 episodes. NOTE: out-of-task-distribution baseline for
# Cosmos (see run_robocasa_cosmos_eval.py's docstring) -- report accordingly.
#
# SLURM ARRAY (one array index per task): ~2.9 min/episode measured
# previously means 450 episodes serially in one job would be ~22 hours.
# Splitting into 15 independent per-task shards (30 trials each) lets them
# run in parallel instead of serially, and a crash in one shard doesn't lose
# the other 14. Each shard writes to its own results subdir so parallel
# writers never race on one episodes.json.
#
# Carries over the SIGABRT hardening from smoke_cosmos_robocasa.slurm
# (PYTHONUNBUFFERED + core dumps): the smoke crash happened deep into a run
# after many successful inference calls, and these 30-trial shards spend far
# longer in the same "sustained 3-camera MuJoCo/EGL" regime than the
# 6-episode smoke test did, so a repeat is plausible -- treat any shard whose
# episodes.json is missing or short as a crash, not just nonzero exit.
#SBATCH --job-name=oopsieverse_cosmos_fullsuite
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=63000M
#SBATCH --time=05:00:00
#SBATCH --array=0-14
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
EVAL_DIR="$REPO/.worktrees/oopsieverse-eval/oopsieverse_eval"
# Distinct port band from cosmos paper-match (8550-8553) and pi05 (8201, 8500+).
PORT=$((8600 + SLURM_ARRAY_TASK_ID))
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/cosmos_server_fullsuite_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

# shelve_item excluded -- needs a camera-calibration file OopsieVerse doesn't
# ship. Must exactly match ALL_TASKS in run_robocasa_cosmos_eval.py.
TASKS=(pick_egg serve_pastry open_single_door turn_on_faucet turn_on_microwave turn_on_stove
       open_drawer close_drawer place_plate counter_to_microwave prepare_coffee
       prepare_breakfast dishes_to_sink nav_lift_bowl wipe_counter)
TASK_NAME="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Cosmos-Policy full-suite eval: task=$TASK_NAME (array idx $SLURM_ARRAY_TASK_ID), port=$PORT ==="

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

module load anaconda3

bash "$EVAL_DIR/servers/launch_cosmos_server.sh" "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for server on port $PORT..."
MAX_WAIT=1200
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
export PYTHONUNBUFFERED=1
ulimit -c unlimited
CORE_DIR="/ocean/projects/cis250185p/asingal/slurm_logs/cores_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$CORE_DIR"
cd "$CORE_DIR"

python -u "$EVAL_DIR/eval/run_robocasa_cosmos_eval.py" \
    --tasks "$TASK_NAME" \
    --num_trials 30 \
    --host localhost --port "$PORT" \
    --results_dir "$EVAL_DIR/results/cosmos_robocasa_fullsuite/$TASK_NAME" \
    --seed 7

EXIT=$?
echo "=== $TASK_NAME done, exit=$EXIT ==="
exit $EXIT
