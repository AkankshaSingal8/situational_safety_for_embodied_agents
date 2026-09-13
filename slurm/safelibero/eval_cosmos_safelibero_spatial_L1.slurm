#!/bin/bash
#SBATCH --job-name=cosmos_spatial_L1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=63000M
#SBATCH --time=05:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
COSMOS_DIR=$REPO/cosmos-policy
SIF_PATH=$COSMOS_DIR/cosmos_policy.sif

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p $REPO/cosmos_benchmark/logs
mkdir -p $REPO/cosmos_video

# ── Verify SIF exists ──────────────────────────────────────────────────────────
if [ ! -f "$SIF_PATH" ]; then
    echo "ERROR: SIF not found at $SIF_PATH"
    echo "Run slurm/build_cosmos_sif.slurm first to build the container."
    exit 1
fi

# ── Run evaluation inside Apptainer container ──────────────────────────────────
# --nv: pass through NVIDIA GPU to container
# --bind: mount host paths into container (same absolute paths, no code changes needed)
# MUJOCO_GL=egl: headless rendering on HPC
# UV_PROJECT_ENVIRONMENT: use the existing .venv_rhel8 directly — its binaries were
#   compiled for manylinux/glibc>=2.28; the container's Ubuntu 24.04 (glibc 2.39)
#   satisfies all requirements including transformer_engine's glibc 2.29 requirement.

apptainer exec --nv \
    --env MUJOCO_GL=egl \
    --env HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache \
    --env TRANSFORMERS_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache \
    --env HF_TOKEN=$(cat /jet/home/asingal/.cache/huggingface/token 2>/dev/null || echo "") \
    --bind $REPO:$REPO \
    --bind /ocean/projects/cis250185p/asingal/.hf_cache:/ocean/projects/cis250185p/asingal/.hf_cache \
    --bind /jet/home/asingal:/jet/home/asingal \
    $SIF_PATH \
    bash -c "$COSMOS_DIR/.venv_rhel8/bin/python \
        $REPO/vlm_pipeline/run_safelibero_cosmos_policy_eval.py \
            --task_suite_name safelibero_spatial \
            --safety_level I \
            --num_trials_per_task 50 \
            --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
            --results_output_dir $REPO/cosmos_benchmark \
            --video_output_dir $REPO/cosmos_video \
            --local_log_dir $REPO/cosmos_benchmark/logs \
            --seed 195 \
            --deterministic True \
            --use_wandb False"
