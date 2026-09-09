#!/bin/bash
#SBATCH --job-name=fastwam_bootstrap
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

set -euo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
ENV_PREFIX=/ocean/projects/cis250185p/asingal/envs/fastwam_env
CKPT_DIR=$REPO/fast-wam/checkpoints
FASTWAM_REPO=$REPO/fast-wam/FastWAM
HF_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p $CKPT_DIR
mkdir -p /tmp/fastwam_smoke

# FastWAM's DiffSynth downloader reads these env vars.
# DIFFSYNTH_DOWNLOAD_SOURCE=huggingface  → use HuggingFace instead of ModelScope
#   (ModelScope times out from Bridges2 GPU nodes — it's a Chinese CDN)
# DIFFSYNTH_MODEL_BASE_PATH              → persistent local cache for all DiffSynth files
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export DIFFSYNTH_MODEL_BASE_PATH=$CKPT_DIR

export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export MUJOCO_GL=egl
export DS_BUILD_OPS=0

source $(conda info --base)/etc/profile.d/conda.sh

# ── Step 1: Create fastwam_env if missing ─────────────────────────────────────
if [ ! -d "$ENV_PREFIX" ]; then
    echo "=========================================="
    echo " STEP 1: Creating fastwam_env (fresh)"
    echo "=========================================="
    conda create -p "$ENV_PREFIX" python=3.10 -y
    conda activate "$ENV_PREFIX"
    python -m pip install --upgrade pip setuptools wheel

    pip install \
        torch==2.7.1+cu128 \
        torchvision==0.22.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

    set +e
    pip install torchcodec==0.5 --index-url https://download.pytorch.org/whl/cu128 2>/dev/null \
        || echo "WARNING: torchcodec not available — skipping"
    set -e

    DS_BUILD_OPS=0 pip install \
        accelerate==1.12.0 \
        av==16.0.1 \
        datasets==3.6.0 \
        deepspeed==0.18.5 \
        einops==0.8.1 \
        huggingface-hub==0.29.2 \
        hydra-core==1.3.2 \
        imageio==2.37.0 \
        imageio-ffmpeg==0.6.0 \
        jsonlines==4.0.0 \
        numpy==1.26.4 \
        omegaconf==2.3.0 \
        packaging==25.0 \
        pandas==2.2.3 \
        pillow==12.0.0 \
        regex==2025.11.3 \
        rich==14.2.0 \
        safetensors==0.5.3 \
        termcolor==2.5.0 \
        tqdm==4.66.5 \
        transformers==4.49.0 \
        typing-extensions==4.15.0 \
        wandb==0.23.1 \
        opencv-python-headless \
        scipy \
        matplotlib

    DS_BUILD_OPS=0 pip install -e "$FASTWAM_REPO"

    SAFELIBERO_REPO=$REPO/SafeLIBERO/safelibero
    set +e
    pip install robosuite==1.4.1 bddl==1.0.1 easydict==1.9 future==0.18.2 cloudpickle==2.1.0 gym==0.25.2
    pip install -e "$SAFELIBERO_REPO"
    set -e

    echo "fastwam_env created."
else
    echo "fastwam_env already exists — skipping creation."
    conda activate "$ENV_PREFIX"
fi

conda activate "$ENV_PREFIX"

# ── Step 1b: Repair / ensure required packages (runs every time) ──────────────
# robosuite upgrades numpy to 2.x; fastwam requires 1.26.4.
# matplotlib is required by SafeLIBERO env_wrapper.py.
echo "=========================================="
echo " STEP 1b: Ensuring critical package versions"
echo "=========================================="
pip install --quiet --upgrade matplotlib
pip install --quiet "numpy==1.26.4"
pip install --quiet sentencepiece protobuf

echo "Package versions:"
python -c "import numpy; print('  numpy:', numpy.__version__)"
python -c "import matplotlib; print('  matplotlib:', matplotlib.__version__)"
python -c "import torch; print('  torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import fastwam; print('  fastwam: OK')"
python -c "from libero.libero.envs import OffScreenRenderEnv; print('  libero OffScreenRenderEnv: OK')"

# ── Step 2: Download checkpoints and DiffSynth model files ────────────────────
echo "=========================================="
echo " STEP 2: Downloading / staging model files"
echo "=========================================="

HF_TOKEN=$(cat /jet/home/asingal/.cache/huggingface/token 2>/dev/null || echo "")

# 2a. Fast-WAM LIBERO checkpoint
if [ ! -f "$CKPT_DIR/libero_uncond_2cam224.pt" ]; then
    echo "  Downloading libero_uncond_2cam224.pt ..."
    huggingface-cli download yuanty/fastwam \
        libero_uncond_2cam224.pt \
        libero_uncond_2cam224_dataset_stats.json \
        --repo-type model \
        --local-dir $CKPT_DIR \
        ${HF_TOKEN:+--token "$HF_TOKEN"}
else
    echo "  libero_uncond_2cam224.pt already present."
fi

# 2b. DiffSynth VAE + text encoder (safetensors)
# FastWAM's loader redirects Wan2.2_VAE.pth → DiffSynth-Studio safetensors.
# Normally it would download these from ModelScope, which times out on Bridges2.
# We pre-download from HuggingFace and set DIFFSYNTH_MODEL_BASE_PATH so the
# loader finds them and skips the runtime download entirely.
DIFFSYNTH_DIR=$CKPT_DIR/DiffSynth-Studio/Wan-Series-Converted-Safetensors
mkdir -p $DIFFSYNTH_DIR

# DiffSynth-Studio/Wan-Series-Converted-Safetensors exists only on ModelScope
# (not HuggingFace), and ModelScope times out from Bridges2 GPU nodes.
# We already have both files as .pth from Wan-AI/Wan2.2-TI2V-5B on HuggingFace.
# Convert them to safetensors in-place — FastWAM's hash check uses tensor key
# names/shapes, not file bytes, so the converted files pass the registry check.
WAN22_SNAP=$HF_CACHE/models--Wan-AI--Wan2.2-TI2V-5B/snapshots
WAN22_HASH=$(ls $WAN22_SNAP | head -1)
VAE_PTH=$(readlink -f $WAN22_SNAP/$WAN22_HASH/Wan2.2_VAE.pth)
T5_PTH=$(readlink -f $WAN22_SNAP/$WAN22_HASH/models_t5_umt5-xxl-enc-bf16.pth)

python - <<PYEOF
import os, sys, torch
from safetensors.torch import save_file

diffsynth_dir = "$DIFFSYNTH_DIR"
os.makedirs(diffsynth_dir, exist_ok=True)

def convert(src, dst):
    if os.path.exists(dst):
        print(f"  Already exists: {dst}")
        return
    print(f"  Converting {os.path.basename(src)} -> {os.path.basename(dst)} ...")
    try:
        sd = torch.load(src, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(src, map_location="cpu", weights_only=False)
    # Unwrap single-key wrappers (mirrors load_state_dict_from_bin)
    if isinstance(sd, dict) and len(sd) == 1:
        k = next(iter(sd))
        if k in ("state_dict", "module", "model_state"):
            sd = sd[k]
    # safetensors requires contiguous float tensors only
    sd = {k: v.contiguous() for k, v in sd.items() if isinstance(v, torch.Tensor)}
    save_file(sd, dst)
    print(f"  Saved {len(sd)} tensors -> {os.path.getsize(dst)/1e9:.2f} GB")

convert("$VAE_PTH", os.path.join(diffsynth_dir, "Wan2.2_VAE.safetensors"))
convert("$T5_PTH",  os.path.join(diffsynth_dir, "models_t5_umt5-xxl-enc-bf16.safetensors"))
PYEOF

ls -lh $DIFFSYNTH_DIR/

# 2c. Tokenizer (Wan-AI/Wan2.1-T2V-1.3B / google/umt5-xxl)
# The tokenizer from Wan2.2-TI2V-5B is identical — reuse it by copying to
# the path FastWAM's loader expects under DIFFSYNTH_MODEL_BASE_PATH.
TOKENIZER_DST=$CKPT_DIR/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl
if [ ! -d "$TOKENIZER_DST" ]; then
    echo "  Staging UMT5-XXL tokenizer from Wan2.2 HF cache ..."
    WAN22_SNAP=$HF_CACHE/models--Wan-AI--Wan2.2-TI2V-5B/snapshots
    WAN22_HASH=$(ls $WAN22_SNAP | head -1)
    TOKENIZER_SRC=$WAN22_SNAP/$WAN22_HASH/google/umt5-xxl
    if [ -d "$TOKENIZER_SRC" ]; then
        mkdir -p $(dirname $TOKENIZER_DST)
        cp -rL $TOKENIZER_SRC $TOKENIZER_DST  # -L dereferences symlinks → copies actual files
        echo "  Tokenizer staged: $(ls $TOKENIZER_DST)"
    else
        echo "  Tokenizer source not found in HF cache — downloading from HF ..."
        mkdir -p $(dirname $TOKENIZER_DST)
        huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
            --include "google/umt5-xxl/**" \
            --repo-type model \
            --local-dir $CKPT_DIR/Wan-AI/Wan2.1-T2V-1.3B \
            ${HF_TOKEN:+--token "$HF_TOKEN"}
    fi
else
    echo "  Tokenizer already staged."
fi

# 2d. Wan2.2-TI2V-5B base model (for DiT weights — needed if not using skip_dit_load)
if [ ! -d "$HF_CACHE/models--Wan-AI--Wan2.2-TI2V-5B" ]; then
    echo "  Downloading Wan2.2-TI2V-5B base model (~22GB) ..."
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
        --repo-type model \
        --cache-dir $HF_CACHE \
        ${HF_TOKEN:+--token "$HF_TOKEN"}
else
    echo "  Wan2.2-TI2V-5B already in HF cache."
fi

echo "  Checkpoint summary:"
ls -lh $CKPT_DIR/*.pt $CKPT_DIR/*.json 2>/dev/null || true
echo "  DiffSynth: $(ls $DIFFSYNTH_DIR 2>/dev/null | tr '\n' ' ')"
echo "  Tokenizer: $(ls $TOKENIZER_DST 2>/dev/null | tr '\n' ' ')"

# ── Step 3: Smoke test (task 0, 1 episode) ────────────────────────────────────
echo "=========================================="
echo " STEP 3: Smoke test (task 0, 1 episode)"
echo "=========================================="

export PYTHONPATH=$REPO/vlm_pipeline:$REPO/SafeLIBERO/safelibero:$FASTWAM_REPO/src:$FASTWAM_REPO/experiments:${PYTHONPATH:-}

python $REPO/vlm_pipeline/run_safelibero_fastwam_eval.py \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 1 \
    --task_indices 0 \
    --ckpt_path $CKPT_DIR/libero_uncond_2cam224.pt \
    --stats_path $CKPT_DIR/libero_uncond_2cam224_dataset_stats.json \
    --results_output_dir /tmp/fastwam_smoke \
    --seed 195 \
    --replan_steps 10 \
    --action_horizon 32 \
    --num_inference_steps 20 \
    --gpu_id 0

echo "=========================================="
echo " Smoke test result:"
echo "=========================================="
find /tmp/fastwam_smoke -name "*.json" | head -1 | xargs python -m json.tool 2>/dev/null || \
    find /tmp/fastwam_smoke -name "*.json" | xargs cat 2>/dev/null || \
    echo "No result JSON found"
