#!/bin/bash
# Cosmos-Policy risk-gate spike (plan Task 9) -- NOT a full eval, just
# confirms get_model()/get_action() run cleanly inside the existing
# container with a novel (uncached) instruction and OopsieVerse-shaped
# synthetic obs, with NO sim construction in this process at all.
#
# Deliberately mirrors the EXACT working apptainer invocation from
# slurm/eval_cosmos_safelibero_spatial_L1.slurm (env vars + binds only,
# reusing the container's own glibc for transformer_engine) -- does NOT
# add the extra host-lib bind/LD_LIBRARY_PATH overrides that caused the
# (still-unresolved) segfault in .worktrees/libero-safety-benchmark's
# Cosmos-on-LIBERO-Safety port. If this still segfaults, the LD_LIBRARY_PATH
# theory from that worktree's postmortem was incomplete -- stop and report
# back rather than debugging further under this SLURM job.
#SBATCH --job-name=oopsieverse_cosmos_risk_gate
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=00:45:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

set -uo pipefail
REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
COSMOS_DIR=$REPO/cosmos-policy
SIF_PATH=$COSMOS_DIR/cosmos_policy.sif
SPIKE_SCRIPT=$REPO/.worktrees/oopsieverse-eval/oopsieverse_eval/eval/cosmos_risk_gate_spike.py

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

if [ ! -f "$SIF_PATH" ]; then
    echo "ERROR: SIF not found at $SIF_PATH -- this spike reuses the existing"
    echo "container read-only and does not build one."
    exit 1
fi

apptainer exec --nv \
    --env MUJOCO_GL=egl \
    --env HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache \
    --env TRANSFORMERS_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache \
    --env HF_TOKEN=$(cat /jet/home/asingal/.cache/huggingface/token 2>/dev/null || echo "") \
    --bind $REPO:$REPO \
    --bind /ocean/projects/cis250185p/asingal/.hf_cache:/ocean/projects/cis250185p/asingal/.hf_cache \
    --bind /jet/home/asingal:/jet/home/asingal \
    $SIF_PATH \
    bash -c "$COSMOS_DIR/.venv_rhel8/bin/python $SPIKE_SCRIPT"

EXIT=$?
echo "=== Risk-gate spike exit code: $EXIT ==="
exit $EXIT
