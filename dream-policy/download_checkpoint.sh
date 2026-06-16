#!/usr/bin/env bash
# Downloads DreamZero-AgiBot base model (~56GB) and DreamZero-LIBERO-LoRA adapter (217MB).
# Run on a login node (no GPU needed). Expect 1-3 hours on PSC network.
#
# Usage: bash dream-policy/download_checkpoint.sh
#
# To run in background:
#   nohup bash dream-policy/download_checkpoint.sh \
#     > /ocean/projects/cis250185p/asingal/slurm_logs/dreamzero_download.log 2>&1 &
#   echo "PID: $!"

set -euo pipefail

HF_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache
HF_TOKEN_FILE=/jet/home/asingal/.cache/huggingface/token

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

# Load token if available (needed for gated models)
if [ -f "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
    echo "HuggingFace token loaded from $HF_TOKEN_FILE"
else
    echo "WARNING: No HF token found at $HF_TOKEN_FILE — proceeding without auth."
    unset HF_TOKEN
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /ocean/projects/cis250185p/asingal/envs/dreamzero_env

mkdir -p "$HF_CACHE/hub"

echo "=== Downloading DreamZero-AgiBot base model (~56GB) ==="
huggingface-cli download GEAR-Dreams/DreamZero-AgiBot \
    --repo-type model \
    --local-dir "$HF_CACHE/hub/GEAR-Dreams/DreamZero-AgiBot" \
    ${HF_TOKEN:+--token "$HF_TOKEN"}

echo "=== Downloading DreamZero-LIBERO-LoRA adapter (~217MB) ==="
huggingface-cli download KyleZ0906/DreamZero-LIBERO-LoRA \
    --repo-type model \
    --local-dir "$HF_CACHE/hub/KyleZ0906/DreamZero-LIBERO-LoRA" \
    ${HF_TOKEN:+--token "$HF_TOKEN"}

echo "=== Download complete. Sizes: ==="
du -sh "$HF_CACHE/hub/GEAR-Dreams/DreamZero-AgiBot" 2>/dev/null || echo "AgiBot: not found"
du -sh "$HF_CACHE/hub/KyleZ0906/DreamZero-LIBERO-LoRA" 2>/dev/null || echo "LoRA: not found"

echo ""
echo "=== File listing ==="
echo "--- AgiBot base ---"
ls -lh "$HF_CACHE/hub/GEAR-Dreams/DreamZero-AgiBot"/*.safetensors 2>/dev/null \
    || ls -lh "$HF_CACHE/hub/GEAR-Dreams/DreamZero-AgiBot"/*.bin 2>/dev/null \
    || echo "(no weight files found)"

echo "--- LIBERO LoRA ---"
ls -lh "$HF_CACHE/hub/KyleZ0906/DreamZero-LIBERO-LoRA"/*.safetensors 2>/dev/null \
    || echo "(no weight files found)"
