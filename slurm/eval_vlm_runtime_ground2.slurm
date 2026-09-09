#!/bin/bash -l
#SBATCH --job-name=vlm_rtg2
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=5
#SBATCH --partition=GPU-shared --gres=gpu:v100-32:1
#SBATCH --time=04:00:00 --mem=63000M
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/vlm_rtg2_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/vlm_rtg2_%j.err

# Offline VLM obstacle-grounding accuracy: 3 prompt variants x 2 models
# (Qwen2.5-VL-3B, Qwen2.5-VL-7B 4-bit) on 21 saved task_0 episodes.
# No rollouts — measures Prompt A accuracy against GT obstacle labels.

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$REPO/.worktrees/fol-safety-filter"
OUT="$WT/fol_grounding_accuracy"
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs "$OUT"

command -v module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null
module load anaconda3
conda activate openvla_libero_merged
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export QWEN_SERVER_URL=http://localhost:5010

run_model () {
    local MODEL=$1; local EXTRA=$2; local TAG=$3
    echo "=== Starting Qwen $MODEL $EXTRA ==="
    conda run --no-capture-output -n qwen \
        python "$REPO/vlm_pipeline/qwen_vlm_server.py" \
        --port 5010 --model "$MODEL" $EXTRA \
        > "$OUT/qwen_${TAG}.log" 2>&1 &
    local PID=$!
    local READY=0
    for i in $(seq 1 300); do
        curl -sf http://localhost:5010/health > /dev/null 2>&1 && { READY=1; break; }
        sleep 5
    done
    if [ "$READY" -eq 0 ]; then echo "ERROR: $MODEL server no start"; kill $PID; return 1; fi
    echo "$MODEL ready after $((i*5))s"

    python "$WT/vlm_pipeline/vlm_runtime_ground_validation.py" \
        --episodes_dir /ocean/projects/cis250185p/asingal/vlm_inputs/safelibero_spatial \
        --output_json "$OUT/runtime_ground2_${TAG}.json" \
        --img_res 512
    local RC=$?
    # conda run wraps the python server — kill the whole process tree, then
    # wait until the port is actually dead (else next phase reuses the old model).
    pkill -P $PID 2>/dev/null; kill $PID 2>/dev/null
    pkill -f "qwen_vlm_server.py --port 5010" 2>/dev/null
    for i in $(seq 1 30); do
        curl -sf http://localhost:5010/health > /dev/null 2>&1 || break
        sleep 2
    done
    if curl -sf http://localhost:5010/health > /dev/null 2>&1; then
        echo "ERROR: old server still alive on 5010"; return 1
    fi
    echo "port 5010 confirmed dead"
    sleep 10   # let VRAM free
    return $RC
}

run_model qwen2.5-vl-3b "" 3b


echo "=== Done: $(date) ==="
