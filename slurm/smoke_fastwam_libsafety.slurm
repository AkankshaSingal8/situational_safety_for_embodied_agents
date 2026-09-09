#!/bin/bash
#SBATCH --job-name=fastwam_libsafety_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=00:45:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

# Smoke test: Fast-WAM on LIBERO-Safety, 1 episode / 1 task, to verify
# the ported env/task-suite layer works end-to-end against the reused
# (main-repo) fastwam_env conda env + checkpoints. See
# docs/superpowers/specs/2026-07-12-cosmos-fastwam-libsafety-design.md

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE=$REPO/.worktrees/libero-safety-benchmark
CKPT_DIR=$REPO/fast-wam/checkpoints

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p $WORKTREE/fastwam_benchmark_libsafety
mkdir -p $WORKTREE/fastwam_video_libsafety

# ── Conda environment (shared, reused read-only) ────────────────────────────
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250185p/asingal/envs/fastwam_env

# ── Environment variables ────────────────────────────────────────────────────
export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache
export DIFFSYNTH_MODEL_BASE_PATH=$CKPT_DIR
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export LIBERO_CONFIG_PATH=$WORKTREE/LIBERO-Safety/.libero_config

cd $WORKTREE

# ── Run smoke test ────────────────────────────────────────────────────────────
python $WORKTREE/run_libsafety_fastwam_eval.py \
    --task_suite_name human_safety \
    --task_indices 0 \
    --num_trials_per_task 1 \
    --ckpt_path $CKPT_DIR/libero_uncond_2cam224.pt \
    --stats_path $CKPT_DIR/libero_uncond_2cam224_dataset_stats.json \
    --results_output_dir $WORKTREE/fastwam_benchmark_libsafety \
    --video_output_dir $WORKTREE/fastwam_video_libsafety \
    --save_videos \
    --seed 195 \
    --replan_steps 10 \
    --action_horizon 32 \
    --num_inference_steps 20 \
    --gpu_id 0
