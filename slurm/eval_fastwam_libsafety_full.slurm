#!/bin/bash
#SBATCH --job-name=fastwam_libsafety_full
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=18:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

# Full evaluation: Fast-WAM on all 4 LIBERO-Safety suites (human_safety,
# obstacle_avoidance, obstacle_avoidance_human, affordance), 15 tasks/suite x
# 3 trials/task = 45 episodes/suite, matching the OpenVLA/OFT convention
# already run on this benchmark. See
# docs/superpowers/specs/2026-07-12-cosmos-fastwam-libsafety-design.md
# Smoke test (job 42146942) passed: TSR=1.0, CAR=1.0 on human_safety task 0.

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

# ── Run evaluation across all 4 LIBERO-Safety suites ────────────────────────
for SUITE in human_safety obstacle_avoidance obstacle_avoidance_human affordance; do
    echo "=== Starting suite: $SUITE ==="
    python $WORKTREE/run_libsafety_fastwam_eval.py \
        --task_suite_name $SUITE \
        --num_trials_per_task 3 \
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
    echo "=== Finished suite: $SUITE (exit $?) ==="
done
