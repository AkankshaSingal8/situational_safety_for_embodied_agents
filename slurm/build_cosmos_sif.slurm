#!/bin/bash
#SBATCH --job-name=build_cosmos_sif
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=02:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
COSMOS_DIR=$REPO/cosmos-policy
SIF_PATH=$COSMOS_DIR/cosmos_policy.sif

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

echo "=== Step 1: Build Apptainer SIF ==="
echo "Base image: nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 (Ubuntu 24.04, glibc 2.39)"
echo "Output: $SIF_PATH"

cd $COSMOS_DIR

apptainer build --fakeroot $SIF_PATH cosmos_policy.def

echo "=== Step 2: Verify SIF built correctly ==="
apptainer exec $SIF_PATH ldd --version 2>&1 | head -2
apptainer exec $SIF_PATH uv --version

echo "=== Step 3: Rebuild uv venv inside container ==="
# The existing .venv was built on RHEL 8 (glibc 2.28) — rename it as backup
if [ -d "$COSMOS_DIR/.venv" ]; then
    echo "Renaming existing .venv (RHEL8 build) to .venv_rhel8"
    mv $COSMOS_DIR/.venv $COSMOS_DIR/.venv_rhel8
fi

# UV_LINK_MODE=copy: Apptainer bind-mounts may not support hardlinks
# Run uv sync inside the container — creates .venv with Ubuntu 24.04 binaries
apptainer exec --nv \
    --env UV_LINK_MODE=copy \
    --env HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache \
    --bind $REPO:$REPO \
    --bind /ocean/projects/cis250185p/asingal/.hf_cache:/ocean/projects/cis250185p/asingal/.hf_cache \
    $SIF_PATH \
    bash -c "cd $COSMOS_DIR && uv sync --extra cu128 --group libero --python 3.10"

echo "=== Step 4: Verify transformer_engine loads inside container ==="
apptainer exec --nv \
    --env UV_LINK_MODE=copy \
    --bind $REPO:$REPO \
    $SIF_PATH \
    bash -c "cd $COSMOS_DIR && uv run --extra cu128 --group libero --python 3.10 python -c 'import transformer_engine; print(\"transformer_engine OK\")'"

echo "=== Build complete. SIF at: $SIF_PATH ==="
