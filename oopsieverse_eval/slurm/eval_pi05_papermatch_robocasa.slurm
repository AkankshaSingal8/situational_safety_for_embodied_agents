#!/bin/bash
# pi0.5 RoboCasa eval, paper-matched protocol: the exact 4-task subset and
# 30-trial count OopsieVerse's own paper uses (Sec V-A/V-C, Appendix E), so
# results here are directly comparable to the paper's published numbers
# (GR00T: 73% completion / 18% safe-completion; base pi0: 17.5% / 0%).
# Single job, not an array: 4 tasks x 30 = 120 episodes, and pi0.5's smoke
# test showed well under a minute/episode -- cheap enough that one server
# instance amortizing its startup (JAX/XLA JIT + first-call compile) once
# beats splitting into shards.
#SBATCH --job-name=oopsieverse_pi05_papermatch
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail
echo "=== OopsieVerse RoboCasa x pi0.5 PAPER-MATCH eval (4 tasks x 30 trials) ==="

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval
EVAL_DIR="$REPO/oopsieverse_eval"
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/pi05_server_papermatch_${SLURM_JOB_ID}.log
PORT=8201

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
    --tasks open_single_door turn_on_microwave counter_to_microwave turn_on_stove \
    --num_trials 30 \
    --host localhost --port "$PORT" \
    --results_dir "$EVAL_DIR/results/pi05_robocasa_papermatch" \
    --seed 7

EXIT=$?
echo "=== Paper-match eval done, exit=$EXIT ==="
exit $EXIT
