#!/bin/bash
# Builds the B1K Apptainer container (Ubuntu 24.04, glibc 2.39) and installs
# BEHAVIOR-1K/OmniGibson/Isaac Sim into a fresh `oopsieverse_b1k` conda env
# INSIDE it. Mirrors slurm/build_cosmos_sif.slurm's structure (same base
# image family, same --fakeroot build, same "rebuild inside the container"
# reasoning) but for BEHAVIOR-1K/OmniGibson instead of cosmos-policy/uv.
#
# Why this exists: bare-metal install on this node's RHEL 8.10 (glibc 2.28)
# fails at Isaac Sim 4.5.0's pip wheel
# (omniverse_kit-...-manylinux_2_34_x86_64.whl needs glibc>=2.34) -- see
# task-1-report.md. This container supplies a newer-glibc userland.
#
# Isolation: only builds oopsieverse_eval/setup/b1k_container.sif and
# creates a NEW `oopsieverse_b1k` conda env (replacing the non-functional
# bare-metal attempt of the same name, which never got a working Isaac Sim
# install). Does not touch oopsieverse_robocasa, oopsieverse_openvla_srv,
# oopsieverse_pi05_srv, oopsieverse_fastwam_srv, or the Cosmos container.
#SBATCH --job-name=build_b1k_sif
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=63000M
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail
REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
EVAL_DIR="$REPO/.worktrees/oopsieverse-eval/oopsieverse_eval"
SETUP_DIR="$EVAL_DIR/setup"
SIF_PATH="$SETUP_DIR/b1k_container.sif"
OOPSIEVERSE_DIR="$EVAL_DIR/external/oopsieverse"
ENV_PATH="/ocean/projects/cis250185p/asingal/envs/oopsieverse_b1k"

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

echo "=== Step 1: Build Apptainer SIF ==="
echo "Base image: nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 (Ubuntu 24.04, glibc 2.39)"
echo "Output: $SIF_PATH"
cd "$SETUP_DIR"
apptainer build --fakeroot "$SIF_PATH" b1k_container.def
BUILD_EXIT=$?
if [ "$BUILD_EXIT" -ne 0 ]; then
    echo "=== SIF build FAILED (exit=$BUILD_EXIT) ==="
    exit "$BUILD_EXIT"
fi

echo "=== Step 2: Verify SIF built correctly ==="
apptainer exec "$SIF_PATH" ldd --version 2>&1 | head -2
apptainer exec "$SIF_PATH" /opt/miniconda3/bin/conda --version

echo "=== Step 3: Remove the non-functional bare-metal oopsieverse_b1k env (Isaac Sim install never succeeded there) ==="
if [ -d "$ENV_PATH" ]; then
    echo "Removing stale bare-metal env at $ENV_PATH"
    rm -rf "$ENV_PATH"
fi

echo "=== Step 4: Install BEHAVIOR-1K/OmniGibson/Isaac Sim inside the container ==="
apptainer exec --nv \
    --bind /ocean/projects/cis250185p/asingal:/ocean/projects/cis250185p/asingal \
    --bind /jet/home/asingal:/jet/home/asingal \
    "$SIF_PATH" \
    bash "$SETUP_DIR/setup_b1k_env_container.sh"
INSTALL_EXIT=$?

echo "=== Step 5: Verify omnigibson imports inside the container ==="
apptainer exec --nv \
    --bind /ocean/projects/cis250185p/asingal:/ocean/projects/cis250185p/asingal \
    --bind /jet/home/asingal:/jet/home/asingal \
    "$SIF_PATH" \
    bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate oopsieverse_b1k && OMNIGIBSON_HEADLESS=1 python -c 'import omnigibson; print(\"omnigibson import OK\")'"
VERIFY_EXIT=$?

echo "=== build_b1k_sif.slurm done. install_exit=$INSTALL_EXIT verify_exit=$VERIFY_EXIT ==="
exit $((INSTALL_EXIT + VERIFY_EXIT))
