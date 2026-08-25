#!/bin/bash
# Correctness check: eager_sample_actions() (correction disabled) must
# match Pi0.sample_actions()'s traced output numerically, before the
# flow-CBF baseline (English et al., arXiv:2607.01378) is trusted for a
# real eval.
#SBATCH --job-name=validate_flow_cbf
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=00:20:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.err

set -uo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark
python validate_flow_cbf.py
