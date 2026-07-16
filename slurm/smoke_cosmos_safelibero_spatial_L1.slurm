#!/bin/bash
#SBATCH --job-name=cosmos_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=63000M
#SBATCH --time=00:30:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

# Smoke test: 2 trials per task on Spatial LI to verify flip-fix is working.
# Expect: episodes complete in <300 steps and TSR > 0%.

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
COSMOS_DIR=$REPO/cosmos-policy
SIF_PATH=$COSMOS_DIR/cosmos_policy.sif

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p $REPO/cosmos_benchmark/logs
mkdir -p $REPO/cosmos_video

if [ ! -f "$SIF_PATH" ]; then
    echo "ERROR: SIF not found at $SIF_PATH"
    exit 1
fi

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
            --num_trials_per_task 2 \
            --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
            --results_output_dir $REPO/cosmos_benchmark \
            --video_output_dir $REPO/cosmos_video \
            --local_log_dir $REPO/cosmos_benchmark/logs \
            --seed 195 \
            --deterministic True \
            --use_wandb False \
            --run_id_note smoke-flip-fix"
