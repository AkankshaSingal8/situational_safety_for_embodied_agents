#!/bin/bash
# GR00T-N1.6 RoboCasa eval, paper-matched protocol: the same 4-task subset
# and 30-trial count as eval_pi05_papermatch_robocasa.slurm /
# eval_cosmos_papermatch_robocasa.slurm (Sec V-A/V-C, Appendix E), so
# results are directly comparable. Single job, not an array: 4 tasks x 30 =
# 120 episodes, one groot_server.py instance amortizing its ~6min model-load
# cost once (per the risk-gate smoke test, job 42911329) beats 4 separate
# shards each re-paying that cost.
#SBATCH --job-name=oopsieverse_groot_papermatch
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail
echo "=== OopsieVerse RoboCasa x GR00T-N1.6 PAPER-MATCH eval (4 tasks x 30 trials) ==="

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval
EVAL_DIR="$REPO/oopsieverse_eval"
GROOT_ENV=/ocean/projects/cis250185p/asingal/envs/oopsieverse_groot_srv
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/groot_server_papermatch_${SLURM_JOB_ID}.log
PORT=8600

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
    --tasks open_single_door turn_on_microwave counter_to_microwave turn_on_stove \
    --num_trials 30 \
    --server_host localhost --server_port "$PORT" \
    --results_dir "$EVAL_DIR/results/groot_robocasa_papermatch" \
    --seed 7

EXIT=$?
echo "=== Paper-match eval done, exit=$EXIT ==="
exit $EXIT
