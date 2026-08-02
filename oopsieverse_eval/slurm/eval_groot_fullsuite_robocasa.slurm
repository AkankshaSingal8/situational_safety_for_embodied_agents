#!/bin/bash
# Full GR00T-N1.6 RoboCasa evaluation: all 15 tasks (shelve_item excluded)
# x 30 trials = 450 episodes.
#
# SLURM ARRAY (one array index per task), following
# eval_pi05_fullsuite_robocasa.slurm's pattern: each shard gets its own
# GPU, its own groot_server.py instance (own port), and writes to its own
# results subdir -- critical: without per-task subdirs, N shards racing on
# one episodes.json would silently keep only 1/N of the results (same
# reasoning as the pi0.5 fullsuite script this mirrors).
#SBATCH --job-name=oopsieverse_groot_fullsuite
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=05:00:00
#SBATCH --array=0-14
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval
EVAL_DIR="$REPO/oopsieverse_eval"
GROOT_ENV=/ocean/projects/cis250185p/asingal/envs/oopsieverse_groot_srv
# Distinct port band from papermatch (8600) and the smoke test (5555).
PORT=$((8700 + SLURM_ARRAY_TASK_ID))
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/groot_server_fullsuite_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

# shelve_item excluded -- needs a camera-calibration file OopsieVerse
# doesn't ship. Must exactly match ALL_TASKS in run_robocasa_groot_eval.py.
TASKS=(pick_egg serve_pastry open_single_door turn_on_faucet turn_on_microwave turn_on_stove
       open_drawer close_drawer place_plate counter_to_microwave prepare_coffee
       prepare_breakfast dishes_to_sink nav_lift_bowl wipe_counter)
TASK_NAME="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== GR00T-N1.6 full-suite eval: task=$TASK_NAME (array idx $SLURM_ARRAY_TASK_ID), port=$PORT ==="

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

module load anaconda3
module load cuda/12.6.1
export HF_HOME=/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/hf

"$GROOT_ENV/bin/python" "$EVAL_DIR/servers/groot_server.py" --port "$PORT" \
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
export PYTHONUNBUFFERED=1

python -u "$EVAL_DIR/eval/run_robocasa_groot_eval.py" \
    --tasks "$TASK_NAME" \
    --num_trials 30 \
    --server_host localhost --server_port "$PORT" \
    --results_dir "$EVAL_DIR/results/groot_robocasa_fullsuite/$TASK_NAME" \
    --seed 7

EXIT=$?
echo "=== $TASK_NAME done, exit=$EXIT ==="
exit $EXIT
