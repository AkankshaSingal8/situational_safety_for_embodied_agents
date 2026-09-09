#!/usr/bin/env bash
# Launches DreamZero WAM inference server on 2 GPUs via torchrun.
#
# PREREQUISITE: Run merge_lora.py first to create the merged checkpoint.
#   conda activate /ocean/projects/cis250185p/asingal/envs/dreamzero_env
#   python dream-policy/merge_lora.py
#
# Usage: bash dream-policy/start_dreamzero_server.sh [PORT]
# Default port: 8000

set -euo pipefail

PORT=${1:-8000}
REPO_ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
DREAMZERO_DIR=$REPO_ROOT/dream-policy/dreamzero
HF_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache
MERGED_CKPT=$HF_CACHE/hub/DreamZero-LIBERO-merged

if [ ! -d "$MERGED_CKPT" ]; then
    echo "ERROR: Merged checkpoint not found at $MERGED_CKPT"
    echo "Run first: conda activate /ocean/projects/cis250185p/asingal/envs/dreamzero_env"
    echo "           python $REPO_ROOT/dream-policy/merge_lora.py"
    exit 1
fi

if [ ! -f "$DREAMZERO_DIR/socket_test_optimized_AR.py" ]; then
    echo "ERROR: DreamZero server script not found at $DREAMZERO_DIR/socket_test_optimized_AR.py"
    echo "Run first: git clone https://github.com/dreamzero0/dreamzero $DREAMZERO_DIR"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /ocean/projects/cis250185p/asingal/envs/dreamzero_env

export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE

echo "=== Starting DreamZero WAM server ==="
echo "    Port:       $PORT"
echo "    Checkpoint: $MERGED_CKPT"
echo "    GPUs:       0, 1 (CUDA_VISIBLE_DEVICES=0,1)"
echo ""

cd "$DREAMZERO_DIR"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    socket_test_optimized_AR.py \
    --port "$PORT" \
    --model-path "$MERGED_CKPT" \
    --enable-dit-cache
