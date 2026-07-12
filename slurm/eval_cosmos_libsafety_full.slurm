#!/bin/bash
#SBATCH --job-name=cosmos_libsafety_full
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=63000M
#SBATCH --time=18:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

# Full evaluation: Cosmos Policy on all 4 LIBERO-Safety suites (human_safety,
# obstacle_avoidance, obstacle_avoidance_human, affordance), 15 tasks/suite x
# 3 trials/task = 45 episodes/suite, matching the OpenVLA/OFT and Fast-WAM
# convention already run on this benchmark. See
# docs/superpowers/specs/2026-07-12-cosmos-fastwam-libsafety-design.md
# Smoke test (job 42157836) passed: TSR=1.0, CAR=1.0 on human_safety task 0,
# using the scoped-MagickWand-library fix (see slurm/smoke_cosmos_libsafety.slurm
# for the full incident writeup: LD_LIBRARY_PATH must NOT include the whole
# host /usr/lib64, only a small scoped symlink dir).

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE=$REPO/.worktrees/libero-safety-benchmark
COSMOS_DIR=$REPO/cosmos-policy
SIF_PATH=$COSMOS_DIR/cosmos_policy.sif

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p $WORKTREE/cosmos_benchmark_libsafety/logs
mkdir -p $WORKTREE/cosmos_video_libsafety

if [ ! -f "$SIF_PATH" ]; then
    echo "ERROR: SIF not found at $SIF_PATH"
    exit 1
fi

for SUITE in human_safety obstacle_avoidance obstacle_avoidance_human affordance; do
    echo "=== Starting suite: $SUITE ==="
    apptainer exec --nv \
        --env MUJOCO_GL=egl \
        --env HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache \
        --env TRANSFORMERS_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache \
        --env HF_TOKEN=$(cat /jet/home/asingal/.cache/huggingface/token 2>/dev/null || echo "") \
        --env LIBERO_CONFIG_PATH=$WORKTREE/LIBERO-Safety/.libero_config \
        --env MAGICK_HOME=$COSMOS_DIR/magickwand_home \
        --bind $REPO:$REPO \
        --bind /ocean/projects/cis250185p/asingal/.hf_cache:/ocean/projects/cis250185p/asingal/.hf_cache \
        --bind /jet/home/asingal:/jet/home/asingal \
        --bind /usr/lib64:/host-lib64:ro \
        $SIF_PATH \
        bash -c "cd $WORKTREE && export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$COSMOS_DIR/magickwand_libs\" && $COSMOS_DIR/.venv_rhel8/bin/python \
            $WORKTREE/run_libsafety_cosmos_policy_eval.py \
                --task_suite_name $SUITE \
                --all_tasks True \
                --num_trials_per_task 3 \
                --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
                --results_output_dir $WORKTREE/cosmos_benchmark_libsafety \
                --video_output_dir $WORKTREE/cosmos_video_libsafety \
                --local_log_dir $WORKTREE/cosmos_benchmark_libsafety/logs \
                --seed 195 \
                --deterministic True \
                --run_id_note full"
    echo "=== Finished suite: $SUITE (exit $?) ==="
done
