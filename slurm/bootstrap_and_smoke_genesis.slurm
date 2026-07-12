#!/bin/bash
#SBATCH --job-name=genesis_bootstrap
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
ENV_PATH=/ocean/projects/cis250185p/asingal/envs/genesis_env
SMOKE_DIR=$REPO/genesis_bakeoff/smoke_test
RESULT_LOG=$SMOKE_DIR/smoke_result_${SLURM_JOB_ID}.log

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p $SMOKE_DIR

module load cuda/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"

# ── Step 1: Create genesis_env if missing ─────────────────────────────────────
if [ ! -d "$ENV_PATH" ]; then
    echo "=========================================="
    echo " STEP 1: Creating genesis_env (Python 3.10)"
    echo "=========================================="
    conda create -p "$ENV_PATH" python=3.10 -y
    conda activate "$ENV_PATH"

    echo "--- Installing PyTorch 2.4+ with CUDA 12.x wheels (Genesis dependency) ---"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    echo "--- Installing Genesis ---"
    pip install genesis-world

    echo "--- Verifying GPU access ---"
    python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('CUDA version:', torch.version.cuda)
print('PyTorch version:', torch.__version__)
"
    echo "genesis_env created."
else
    echo "genesis_env already exists — skipping creation."
    conda activate "$ENV_PATH"
fi

conda activate "$ENV_PATH"

# ── Step 2: Headless smoke test ───────────────────────────────────────────────
# Verifies, in order: (1) genesis imports and initializes on this GPU, (2) a
# rigid-body scene builds and steps physics, (3) headless/offscreen RGB+depth
# rendering works without RT cores (V100 has none) — this is the exact
# capability that was verified as blocked for Isaac Sim and NOT blocked for
# Genesis per prior research; this smoke test confirms it live on our actual
# cluster hardware rather than trusting docs alone.
echo "=========================================="
echo " STEP 2: Headless Genesis smoke test"
echo "=========================================="

cat > "$SMOKE_DIR/smoke_test.py" << 'PYEOF'
import sys
import numpy as np

print("--- Importing genesis ---")
import genesis as gs

print("--- Initializing genesis (gpu backend) ---")
gs.init(backend=gs.gpu, logging_level="warning")

print("--- Building minimal rigid-body scene ---")
scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(dt=0.01),
)
plane = scene.add_entity(gs.morphs.Plane())
box = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0, 0, 0.5)))

cam = scene.add_camera(
    res=(256, 256),
    pos=(1.5, 0.0, 1.0),
    lookat=(0.0, 0.0, 0.3),
    fov=45,
    GUI=False,
)

print("--- Building scene (this triggers GPU kernel compilation) ---")
scene.build()

print("--- Stepping physics 50 steps ---")
for i in range(50):
    scene.step()

print("--- Rendering headless RGB + depth frame ---")
rgb, depth, _, _ = cam.render(depth=True)
print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")

final_box_pos = box.get_pos()
print(f"Box final position after 50 steps (should have fallen from z=0.5): {final_box_pos}")

assert rgb.shape[:2] == (256, 256), "RGB render shape mismatch"
assert depth.shape[:2] == (256, 256), "Depth render shape mismatch"
assert float(final_box_pos[2]) < 0.5, "Box did not fall — physics step may not be working"

print("SMOKE_TEST_RESULT: PASS")
PYEOF

python "$SMOKE_DIR/smoke_test.py" 2>&1 | tee "$RESULT_LOG"
SMOKE_EXIT=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo " Genesis smoke test result (exit code: $SMOKE_EXIT)"
echo "=========================================="
grep "SMOKE_TEST_RESULT" "$RESULT_LOG" || echo "SMOKE_TEST_RESULT: FAIL (no result line found)"

exit $SMOKE_EXIT
