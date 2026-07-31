#!/bin/bash
# Full pi0.5 BEHAVIOR-1K (OmniGibson) evaluation: all 13 tasks in this
# fork's registry x 30 trials = 390 episodes.
#
# This covers BOTH deliverables in one campaign, not two: the paper's
# matched 3/4-task subset (add_firewood, wipe_counter, food_in_microwave --
# see run_b1k_pi05_eval.py's PAPER_SUBSET_TASKS comment; "attach camera" has
# no equivalent task in this registry, confirmed against this fork's own
# published task table, so it is not run) is just a filter over this same
# 13-task x 30-trial results set, not a second campaign -- given Isaac
# Sim's ~25-minute cold start per array shard, running the subset
# separately would roughly double GPU time for zero new information.
#
# SLURM ARRAY (one array index per task, one Kit boot per shard -- an
# adapter instance is reused across all 30 trials within a shard, NOT
# reconstructed per trial, see b1k_adapter.py's docstring for why).
# Requests 2 GPUs per shard (server + Isaac Sim client each pinned to
# their own GPU) -- see smoke_b1k_pi05.slurm for the VRAM-contention
# rationale (JAX's XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 vs Isaac Sim's own
# RTX device-creation requirement).
#SBATCH --job-name=oopsieverse_b1k_pi05_fullsuite
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96000M
#SBATCH --time=04:00:00
#SBATCH --array=0-12
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
EVAL_DIR="$REPO/.worktrees/oopsieverse-eval/oopsieverse_eval"
SIF_PATH="$EVAL_DIR/setup/b1k_container.sif"
# Distinct port band from RoboCasa's pi0.5 jobs (8200/8201/8500+N).
PORT=$((8600 + SLURM_ARRAY_TASK_ID))
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/pi05_server_b1k_fullsuite_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

# Must exactly match ALL_TASKS in run_b1k_pi05_eval.py / this fork's
# oopsiebench/envs/behavior1k/ TASK_REGISTRY (minus "default").
TASKS=(add_firewood fill_bowl food_in_microwave heat_saucepot nav_to_table
       open_drawer open_single_door pick_egg place_plate pour_water
       shelve_item turn_on_faucet wipe_counter)
TASK_NAME="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== B1K x pi0.5 full-suite eval: task=$TASK_NAME (array idx $SLURM_ARRAY_TASK_ID), port=$PORT ==="

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

if [ ! -f "$SIF_PATH" ]; then
    echo "ERROR: SIF not found at $SIF_PATH"
    exit 1
fi

module load anaconda3

conda activate /ocean/projects/cis250185p/asingal/envs/oopsieverse_pi05_srv
export OPENPI_DATA_HOME=/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/openpi
# See smoke_b1k_pi05.slurm for why this is lowered from RoboCasa's 0.9
# (Vulkan/Isaac Sim GPU selection may not honor CUDA_VISIBLE_DEVICES).
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export CUDA_VISIBLE_DEVICES=0

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

LD_LIBRARY_PATH_ADDITIONS_CMD='ENVROOT=/ocean/projects/cis250185p/asingal/envs/oopsieverse_b1k/lib/python3.10/site-packages/isaacsim; LDP=$(find "$ENVROOT" -maxdepth 3 -type d -name bin 2>/dev/null | tr "\n" ":"); export LD_LIBRARY_PATH="$LDP:${LD_LIBRARY_PATH:-}"'

ICD_JSON=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
NVIDIA_GLX_LIB=/usr/lib64/libGLX_nvidia.so.0
if [ ! -f "$ICD_JSON" ] || [ ! -f "$NVIDIA_GLX_LIB" ]; then
    echo "ERROR: expected Vulkan ICD file(s) not found on this node."
    exit 1
fi

apptainer exec --nv \
    --bind /ocean/projects/cis250185p/asingal:/ocean/projects/cis250185p/asingal \
    --bind /jet/home/asingal:/jet/home/asingal \
    --bind "$ICD_JSON:$ICD_JSON:ro" \
    --bind "$NVIDIA_GLX_LIB:$NVIDIA_GLX_LIB:ro" \
    --env OMNIGIBSON_HEADLESS=1 \
    --env VK_ICD_FILENAMES="$ICD_JSON" \
    --env CUDA_VISIBLE_DEVICES=1 \
    "$SIF_PATH" \
    bash -c "$LD_LIBRARY_PATH_ADDITIONS_CMD && source /opt/miniconda3/etc/profile.d/conda.sh && conda activate oopsieverse_b1k && python -u '$EVAL_DIR/eval/run_b1k_pi05_eval.py' --tasks $TASK_NAME --num_trials 30 --host localhost --port $PORT --results_dir '$EVAL_DIR/results/pi05_b1k_fullsuite/$TASK_NAME' --seed 7"

EXIT=$?
echo "=== $TASK_NAME done, exit=$EXIT ==="
exit $EXIT
