#!/bin/bash
#SBATCH --job-name=fastwam_spatial_L1
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

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
CKPT_DIR=$REPO/fast-wam/checkpoints

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p $REPO/fastwam_benchmark
mkdir -p $REPO/fastwam_video

# ── Conda environment ──────────────────────────────────────────────────────────
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250185p/asingal/envs/fastwam_env

# ── Environment variables ──────────────────────────────────────────────────────
export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache
# FastWAM's DiffSynth loader: use local pre-converted files, never call ModelScope
# (ModelScope times out from Bridges2; files were converted from .pth in bootstrap job)
export DIFFSYNTH_MODEL_BASE_PATH=$CKPT_DIR
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

# ── Python path ───────────────────────────────────────────────────────────────
export PYTHONPATH=$REPO/vlm_pipeline:$REPO/SafeLIBERO/safelibero:$REPO/fast-wam/FastWAM/src:$REPO/fast-wam/FastWAM/experiments:$PYTHONPATH

# ── Run evaluation ─────────────────────────────────────────────────────────────
python $REPO/vlm_pipeline/run_safelibero_fastwam_eval.py \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 50 \
    --ckpt_path $CKPT_DIR/libero_uncond_2cam224.pt \
    --stats_path $CKPT_DIR/libero_uncond_2cam224_dataset_stats.json \
    --results_output_dir $REPO/fastwam_benchmark \
    --video_output_dir $REPO/fastwam_video \
    --save_videos \
    --seed 195 \
    --replan_steps 10 \
    --action_horizon 32 \
    --num_inference_steps 20 \
    --gpu_id 0
