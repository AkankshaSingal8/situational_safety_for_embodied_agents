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
#
# First attempt (job 42877662) segfaulted at Kit boot:
# "ImportError: libhdx.so: cannot open shared object file" from
# omni.physx/omni.kit.xr.core, even though libhdx.so DOES exist under
# isaacsim/extscache/omni.usd.libs-*/bin/ -- a library-search-path issue,
# not a missing file. Isaac Sim's pip distribution normally relies on each
# extension's own bin/ dir being added to the loader path as extensions
# start up; add every isaacsim extscache/exts*/bin dir to LD_LIBRARY_PATH
# up front so this doesn't depend on load order.
LD_LIBRARY_PATH_ADDITIONS_CMD='ENVROOT=/ocean/projects/cis250185p/asingal/envs/oopsieverse_b1k/lib/python3.10/site-packages/isaacsim; LDP=$(find "$ENVROOT" -maxdepth 3 -type d -name bin 2>/dev/null | tr "\n" ":"); export LD_LIBRARY_PATH="$LDP:${LD_LIBRARY_PATH:-}"; echo "[slurm] LD_LIBRARY_PATH additions: $LDP"'

# Vulkan ICD fix (job 42879166/42879055 failed with
# "[gpu.foundation.plugin] No device could be created" -- Isaac Sim's
# RTX renderer needs a real Vulkan device, not just CUDA). Diagnosed via
# `vulkaninfo` inside the container: apptainer --nv binds the NVIDIA
# driver .so files under /.singularity.d/libs/, but does NOT bind the
# host's Vulkan ICD registration JSON
# (/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json) or the exact absolute
# path that JSON's "library_path" points at (/usr/lib64/libGLX_nvidia.so.0).
# Without these, the container's Vulkan loader only finds the software
# llvmpipe ICD (no ray tracing) -- confirmed by `vulkaninfo --summary`
# showing deviceType=PHYSICAL_DEVICE_TYPE_CPU / deviceName=llvmpipe.
# Binding both files at their exact host paths and setting
# VK_ICD_FILENAMES fixes it (confirmed: vulkaninfo then reports
# deviceType=PHYSICAL_DEVICE_TYPE_DISCRETE_GPU / deviceName=NVIDIA H100).
ICD_JSON=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
NVIDIA_GLX_LIB=/usr/lib64/libGLX_nvidia.so.0
if [ ! -f "$ICD_JSON" ] || [ ! -f "$NVIDIA_GLX_LIB" ]; then
    echo "ERROR: expected Vulkan ICD file(s) not found on this node ($ICD_JSON, $NVIDIA_GLX_LIB)."
    echo "This diagnostic path is specific to Bridges2 H100 nodes seen so far -- if this fires,"
    echo "the exact host path may differ on this node and needs re-diagnosing (see task-1-report.md)."
    exit 1
fi

apptainer exec --nv \
    --bind /ocean/projects/cis250185p/asingal:/ocean/projects/cis250185p/asingal \
    --bind /jet/home/asingal:/jet/home/asingal \
    --bind "$ICD_JSON:$ICD_JSON:ro" \
    --bind "$NVIDIA_GLX_LIB:$NVIDIA_GLX_LIB:ro" \
    --env OMNIGIBSON_HEADLESS=1 \
    --env VK_ICD_FILENAMES="$ICD_JSON" \
    "$SIF_PATH" \
    bash -c "$LD_LIBRARY_PATH_ADDITIONS_CMD && source /opt/miniconda3/etc/profile.d/conda.sh && conda activate oopsieverse_b1k && python -u '$EVAL_DIR/eval/b1k_headless_spike.py' --task_name pick_egg --n_steps 30 --out_dir '$EVAL_DIR/results/b1k_headless_spike'"

EXIT=$?
echo "=== B1K headless spike done, exit=$EXIT ==="
echo "Check output: $EVAL_DIR/results/b1k_headless_spike/"
exit $EXIT
