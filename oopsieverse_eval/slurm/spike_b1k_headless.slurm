#!/bin/bash
# BEHAVIOR-1K / OmniGibson headless-rendering risk gate (Task 1 of the
# enchanted-finding-piglet plan, Phase 5). This is the actual go/no-go
# test: proves Isaac Sim/OmniGibson runs env.reset()/env.step() and
# produces a real camera render under `sbatch` -- i.e. with NO interactive
# display attached -- not just interactively via `srun --pty`.
#
# Runs INSIDE b1k_container.sif (Ubuntu 24.04 / glibc 2.39 Apptainer
# container -- see b1k_container.def and build_b1k_sif.slurm) because
# bare-metal RHEL 8.10 (glibc 2.28) cannot install Isaac Sim 4.5.0's pip
# wheel (see task-1-report.md for the exact error). Uses the
# container-built `oopsieverse_b1k` conda env only. Does NOT touch
# oopsieverse_robocasa, oopsieverse_openvla_srv, oopsieverse_pi05_srv,
# oopsieverse_fastwam_srv, or the Cosmos Apptainer container.
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
SIF_PATH="$EVAL_DIR/setup/b1k_container.sif"

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

if [ ! -f "$SIF_PATH" ]; then
    echo "ERROR: SIF not found at $SIF_PATH -- build it first with"
    echo "  sbatch $EVAL_DIR/slurm/build_b1k_sif.slurm"
    exit 1
fi

# OmniGibson/Isaac Sim headless mechanism (see task-1-report.md for the
# confirmed citation from the installed package's macros.py) -- set the
# env var; b1k_headless_spike.py relies on this being set before
# `import omnigibson`.
apptainer exec --nv \
    --bind /ocean/projects/cis250185p/asingal:/ocean/projects/cis250185p/asingal \
    --bind /jet/home/asingal:/jet/home/asingal \
    --env OMNIGIBSON_HEADLESS=1 \
    "$SIF_PATH" \
    bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate oopsieverse_b1k && python -u '$EVAL_DIR/eval/b1k_headless_spike.py' --task_name pick_egg --n_steps 30 --out_dir '$EVAL_DIR/results/b1k_headless_spike'"

EXIT=$?
echo "=== B1K headless spike done, exit=$EXIT ==="
echo "Check output: $EVAL_DIR/results/b1k_headless_spike/"
exit $EXIT
