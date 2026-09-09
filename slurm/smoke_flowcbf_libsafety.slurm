#!/bin/bash
# Smoke test: neurosymbolic constrained-flow-matching baseline (English et
# al., arXiv:2607.01378) on ONE task of ONE suite, in-process (single
# conda env needs both jax/openpi AND libero/robosuite/mujoco -- installed
# both into libsafety_openpi for this driver, since the correction runs
# inside the denoising loop and can't go through a separate websocket
# server/client split like the other drivers).
#SBATCH --job-name=smoke_flowcbf_libsafety
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=00:30:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.err

set -uo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE=$REPO/.worktrees/libero-safety-benchmark
LIBSAFETY_DIR=$WORKTREE/LIBERO-Safety
SUITE=obstacle_avoidance

echo "=== [job ${SLURM_JOB_ID}] smoke test: flow-CBF baseline, suite=${SUITE} task=0 ==="

source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="${LIBSAFETY_DIR}/.libero_config"

cd "$WORKTREE"
python run_libsafety_eval_flowcbf.py \
    --task_suite_name "$SUITE" \
    --task_index 0 \
    --num_trials_per_task 1 \
    --checkpoint_name flowcbf_pi05_libero_safety \
    --results_out_path "${LIBSAFETY_DIR}/results" \
    --video_out_path "${LIBSAFETY_DIR}/rollouts" \
    --seed 7
CLIENT_EXIT=$?
echo "=== Client exited with code $CLIENT_EXIT ==="
exit $CLIENT_EXIT
