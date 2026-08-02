#!/bin/bash
# GR00T-N1.6 risk-gate spike (high-risk eval spike): builds the fresh
# `oopsieverse_groot_srv` env (if not already built), downloads
# `nvidia/GR00T-N1.6-3B`, and runs groot_risk_gate_spike.py -- NO sim
# construction, just confirms the checkpoint loads with embodiment_tag
# ROBOCASA_PANDA_OMRON and emits non-degenerate actions on synthetic obs.
# GPU type matters here (unlike this repo's other GPU jobs, which mostly
# already request h100-80): job 42911160 landed on a v100-32 node and died
# with "RuntimeError: CUDA error: no kernel image is available for
# execution on the device" -- GR00T's uv.lock-pinned torch==2.7.1+cu128
# build doesn't ship kernels for compute capability 7.0 (V100's), only
# >=7.5 (confirmed via PyTorch's own startup warning: "PyTorch no longer
# supports this GPU ... minimum cuda capability supported is 7.5"). Bridges2
# GPU-shared mixes v100-32/v100-16 (cap 7.0) with h100-80/l40s-48 (cap
# 8.9/9.0) nodes under one partition, per `sinfo -p GPU-shared -o "%N %G"`
# -- explicitly requesting h100-80 (matching eval_cosmos_*/eval_pi05_*
# .slurm in this same directory) avoids landing on an incompatible node.
#SBATCH --job-name=oopsieverse_groot_risk_gate
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=02:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

set -uo pipefail
REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE=$REPO/.worktrees/oopsieverse-eval
ENV_PREFIX=/ocean/projects/cis250185p/asingal/envs/oopsieverse_groot_srv

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

module load anaconda3
module load cuda/12.6.1

if [ ! -d "$ENV_PREFIX" ]; then
    echo "=== Building oopsieverse_groot_srv env (first run) ==="
    bash "$WORKTREE/oopsieverse_eval/setup/setup_groot_server_env.sh"
else
    echo "=== Reusing existing oopsieverse_groot_srv env ==="
fi

# NOTE: ENV_PREFIX is a `uv`-managed venv (see setup_groot_server_env.sh's
# postmortem on why plain conda+pip couldn't install this package), NOT a
# conda env -- `conda activate` on this path would fail/misbehave since
# there's no conda-meta/ here. Source its own venv activate script instead.
source "$ENV_PREFIX/bin/activate"
export HF_HOME=/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/hf

python "$WORKTREE/oopsieverse_eval/eval/groot_risk_gate_spike.py"
EXIT=$?
echo "=== Risk-gate spike exit code: $EXIT ==="
exit $EXIT
