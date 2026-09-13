#!/bin/bash
#SBATCH --job-name=fastwam_libsafety_full_v2
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=20:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

# Full evaluation v2: Fast-WAM on all 4 LIBERO-Safety suites, one task per
# python invocation (instead of --all_tasks in one process), each wrapped in
# a bash-level retry loop.
#
# WHY: the original v1 run (job 42157759, all tasks in one process) and a
# follow-up diagnostic (job 42161922, PYTHONFAULTHANDLER=1) both crashed with
# a native SIGABRT inside robosuite's MuJoCo/EGL rendering pipeline
# (robosuite/utils/binding_utils.py:171 read_pixels -> render -> camera_rgb,
# called from env.step()) on the very first render call. This is NOT a
# Python exception -- it's an uncatchable process-level abort, likely GPU
# VRAM/EGL-context contention on the GPU-shared partition, so a single
# `try/except` around the task loop (which main() already has) cannot save
# the other 14 tasks in that process. Confirmed the crash is unrelated to
# our LIBERO-Safety porting code and unrelated to model size: the same
# model/task/code succeeded in isolation in the original 1-episode smoke
# test (job 42146942).
#
# FIX: run each task as its own subprocess (matching the per-task JSON file
# convention already used for OpenVLA's LIBERO-Safety results --
# human_safety_task0.json, task1.json, ...), retried up to 3x. A crash then
# only costs one task's worth of (model reload + 3 episodes), not the whole
# 45-episode suite. aggregate_libsafety_results.py combines the per-task
# JSONs into one suite summary afterward.
#
# Uses a SEPARATE output directory from the v1 run and the smoke test to
# avoid mixing in stale/partial results.

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE=$REPO/.worktrees/libero-safety-benchmark
CKPT_DIR=$REPO/fast-wam/checkpoints

RESULTS_DIR=$WORKTREE/fastwam_benchmark_libsafety_full
VIDEO_DIR=$WORKTREE/fastwam_video_libsafety_full

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p $RESULTS_DIR
mkdir -p $VIDEO_DIR

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250185p/asingal/envs/fastwam_env

export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache
export DIFFSYNTH_MODEL_BASE_PATH=$CKPT_DIR
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export LIBERO_CONFIG_PATH=$WORKTREE/LIBERO-Safety/.libero_config

cd $WORKTREE

MAX_RETRIES=3
NUM_TASKS=15  # per LIBERO-Safety suite (confirmed via task_suite.n_tasks for all 4 suites)

for SUITE in human_safety obstacle_avoidance obstacle_avoidance_human affordance; do
    mkdir -p $RESULTS_DIR/$SUITE
    for TASK_IDX in $(seq 0 $((NUM_TASKS - 1))); do
        SUCCESS=0
        for ATTEMPT in $(seq 1 $MAX_RETRIES); do
            echo "=== Suite $SUITE  Task $TASK_IDX  Attempt $ATTEMPT/$MAX_RETRIES ==="
            python $WORKTREE/run_libsafety_fastwam_eval.py \
                --task_suite_name $SUITE \
                --task_indices $TASK_IDX \
                --num_trials_per_task 3 \
                --ckpt_path $CKPT_DIR/libero_uncond_2cam224.pt \
                --stats_path $CKPT_DIR/libero_uncond_2cam224_dataset_stats.json \
                --results_output_dir $RESULTS_DIR \
                --video_output_dir $VIDEO_DIR \
                --save_videos \
                --seed 195 \
                --replan_steps 10 \
                --action_horizon 32 \
                --num_inference_steps 20 \
                --gpu_id 0
            EXIT=$?
            if [ $EXIT -eq 0 ]; then
                SUCCESS=1
                echo "=== Suite $SUITE  Task $TASK_IDX  succeeded on attempt $ATTEMPT ==="
                break
            else
                echo "=== Suite $SUITE  Task $TASK_IDX  attempt $ATTEMPT FAILED (exit $EXIT) ==="
                sleep 15
            fi
        done
        if [ $SUCCESS -eq 0 ]; then
            echo "GAVE UP: suite=$SUITE task=$TASK_IDX after $MAX_RETRIES attempts" | tee -a $RESULTS_DIR/failed_tasks.log
        fi
    done
done

echo "=== Aggregating per-task results into suite summaries ==="
python $WORKTREE/aggregate_libsafety_results.py \
    --results_dir $RESULTS_DIR \
    --suites human_safety obstacle_avoidance obstacle_avoidance_human affordance
