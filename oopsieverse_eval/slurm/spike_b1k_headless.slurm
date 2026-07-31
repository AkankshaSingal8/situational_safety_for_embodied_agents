#!/bin/bash
# BEHAVIOR-1K / OmniGibson headless-rendering risk gate (Task 1 of the
# enchanted-finding-piglet plan, Phase 5). This is the actual go/no-go
# test: proves Isaac Sim/OmniGibson runs env.reset()/env.step() and
# produces a real camera render under `sbatch` -- i.e. with NO interactive
# display attached -- not just interactively via `srun --pty`.
#
# Uses the freshly-built, isolated `oopsieverse_b1k` conda env only. Does
# NOT touch oopsieverse_robocasa, oopsieverse_openvla_srv,
# oopsieverse_pi05_srv, oopsieverse_fastwam_srv, or the Cosmos Apptainer
# container.
#SBATCH --job-name=oopsieverse_b1k_headless_spike
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=63000M
#SBATCH --time=00:45:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail
echo "=== OopsieVerse BEHAVIOR-1K headless spike (risk gate) ==="
echo "Host: $(hostname)"
nvidia-smi || echo "WARNING: nvidia-smi failed"

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
EVAL_DIR="$REPO/.worktrees/oopsieverse-eval/oopsieverse_eval"

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate oopsieverse_b1k

# OmniGibson/Isaac Sim headless mechanism (see report for the confirmed
# citation from the installed package's macros.py) -- set both the env var
# and rely on b1k_headless_spike.py also setting omnigibson.macros.gm.HEADLESS
# directly before the sim boots.
export OMNIGIBSON_HEADLESS=1

python -u "$EVAL_DIR/eval/b1k_headless_spike.py" \
    --task_name pick_egg \
    --n_steps 30 \
    --out_dir "$EVAL_DIR/results/b1k_headless_spike"

EXIT=$?
echo "=== B1K headless spike done, exit=$EXIT ==="
echo "Check output: $EVAL_DIR/results/b1k_headless_spike/"
exit $EXIT
