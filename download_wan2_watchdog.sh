#!/bin/bash
# Download Wan2.1-I2V-14B-480P components using huggingface-cli.
# Restarts automatically if any file stalls for >3 minutes.
set -uo pipefail

export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export NO_ALBUMENTATIONS_UPDATE=1

LOG=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/dream_benchmark/logs/wan_download.log
CACHE=$HF_HOME
REPO="Wan-AI/Wan2.1-I2V-14B-480P"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /ocean/projects/cis250185p/asingal/envs/dreamzero_env

get_cache_bytes() {
    python3 -c "
import os, glob
total = 0
for f in glob.glob('$CACHE/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/**/*', recursive=True):
    if os.path.isfile(f):
        total += os.path.getsize(f)
print(total)
" 2>/dev/null || echo 0
}

FILES=(
    "models_t5_umt5-xxl-enc-bf16.pth"
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    "Wan2.1_VAE.pth"
    "diffusion_pytorch_model.safetensors.index.json"
    "diffusion_pytorch_model-00001-of-00007.safetensors"
    "diffusion_pytorch_model-00002-of-00007.safetensors"
    "diffusion_pytorch_model-00003-of-00007.safetensors"
    "diffusion_pytorch_model-00004-of-00007.safetensors"
    "diffusion_pytorch_model-00005-of-00007.safetensors"
    "diffusion_pytorch_model-00006-of-00007.safetensors"
    "diffusion_pytorch_model-00007-of-00007.safetensors"
)

for FILE in "${FILES[@]}"; do
    echo "[$(date)] Downloading: $FILE" >> "$LOG"
    attempt=0
    while true; do
        attempt=$((attempt + 1))
        last_bytes=$(get_cache_bytes)

        huggingface-cli download "$REPO" "$FILE" \
            --cache-dir "$CACHE" \
            --quiet >> "$LOG" 2>&1 &
        DL_PID=$!
        echo "$DL_PID" > /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/dream_benchmark/logs/wan_download.pid

        stall_count=0
        while kill -0 $DL_PID 2>/dev/null; do
            sleep 60
            if ! kill -0 $DL_PID 2>/dev/null; then break; fi
            cur_bytes=$(get_cache_bytes)
            if [ "$cur_bytes" -gt "$last_bytes" ]; then
                stall_count=0
                gb=$(python3 -c "print(f'{$cur_bytes/1e9:.2f}')")
                echo "[$(date)] Progress: ${gb} GB total" >> "$LOG"
                last_bytes=$cur_bytes
            else
                stall_count=$((stall_count + 1))
                echo "[$(date)] Stall #$stall_count on $FILE (attempt $attempt)" >> "$LOG"
                if [ $stall_count -ge 3 ]; then
                    echo "[$(date)] Killing stalled download, restarting $FILE..." >> "$LOG"
                    kill $DL_PID 2>/dev/null
                    break
                fi
            fi
        done

        wait $DL_PID 2>/dev/null
        EXIT=$?
        if [ $EXIT -eq 0 ]; then
            echo "[$(date)] Done: $FILE" >> "$LOG"
            break
        fi
        echo "[$(date)] Failed (exit $EXIT), retry in 5s..." >> "$LOG"
        sleep 5
    done
done

echo "[$(date)] ALL DOWNLOADS COMPLETE" >> "$LOG"
