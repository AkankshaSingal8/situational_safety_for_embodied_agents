#!/bin/bash
#SBATCH --job-name=localizer_train
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Phase 3 (spec 2026-08-11): train the hazard localizer on frames + free
# sim-GT labels from `localizer_data/frames/` (collected by
# slurm/ls_localizer_data.slurm) and run the G-L1 offline eval gate
# (overall held-out median 3D error < 2.5 cm). CPU-tested logic lives in
# vlm_pipeline/hazard_localizer.py / vlm_pipeline/tests/test_hazard_localizer.py;
# this script is the GPU-time driver only. conda env: openvla_libero_merged
# has torch 2.2.0 (checked -- same client env slurm/ls_localizer_data.slurm
# uses to run the eval client that collects the frames); `aegis` also has
# torch (2.7.1) but is reserved for the policy-serving side elsewhere.
#
# `verify` runs FIRST and aborts (nonzero exit, `set -e`) before any GPU
# training time is spent if the camera/pixel projection convention doesn't
# hold empirically on the real collected frames (in-bounds fraction < 0.95)
# -- see the "Camera geometry" comment in hazard_localizer.py for why this
# is a real (not hypothetical) risk worth a hard gate here.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
DATA="$WT/localizer_data/frames"
OUT="$WT/localizer_out"

conda activate openvla_libero_merged

python "$WT/vlm_pipeline/hazard_localizer.py" verify \
  --data "$DATA" --n 50

python "$WT/vlm_pipeline/hazard_localizer.py" train \
  --data "$DATA" --out "$OUT"

python "$WT/vlm_pipeline/hazard_localizer.py" eval \
  --data "$DATA" --model "$OUT"

echo "LOCALIZER_TRAIN_COMPLETE"
