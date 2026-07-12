#!/bin/bash
#SBATCH --job-name=cosmos_libsafety_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=63000M
#SBATCH --time=00:30:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

# Smoke test: Cosmos Policy on LIBERO-Safety, 1 episode / 1 task, to verify
# the ported env/task-suite layer works end-to-end against the reused
# (main-repo) container + venv. See
# docs/superpowers/specs/2026-07-12-cosmos-fastwam-libsafety-design.md

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE=$REPO/.worktrees/libero-safety-benchmark
COSMOS_DIR=$REPO/cosmos-policy
SIF_PATH=$COSMOS_DIR/cosmos_policy.sif

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p $WORKTREE/cosmos_benchmark_libsafety/logs
mkdir -p $WORKTREE/cosmos_video_libsafety

if [ ! -f "$SIF_PATH" ]; then
    echo "ERROR: SIF not found at $SIF_PATH"
    exit 1
fi

# LIBERO-Safety's env_wrapper.py imports `wand` (ImageMagick bindings) at
# module load time. The container has the `wand` python package (installed
# read-only-safe via `uv pip install --no-deps wand` earlier) but not the
# underlying MagickWand C library, which lives on the HOST at
# /usr/lib64/libMagickWand-6.Q16.so.7 (confirmed present). Bind the host lib
# dir to a SEPARATE in-container path (NOT /usr/lib64 — binding directly over
# /usr/lib64 shadows the container's own newer glibc with the host's older
# RHEL8 glibc, breaking bash itself; see incident in job 42146941).
#
# Putting the WHOLE host /usr/lib64 on LD_LIBRARY_PATH (even appended after
# the container's own paths) still segfaults python at interpreter startup,
# before any of our code runs (incident in jobs 42147010 and 42157696) —
# LD_LIBRARY_PATH entries are searched before the container's own ld.so
# cache, so for any library not already found in the container's existing
# LD_LIBRARY_PATH dirs, the host version wins and is ABI-incompatible with
# the container's compiled binaries (confirmed via manual apptainer exec:
# `python -c 'print(...)'` alone segfaults with /host-lib64 on the path).
#
# Fix: only expose the ~12 libraries MagickWand actually needs that are
# missing from the container (confirmed via `ldconfig -p` diff inside the
# container against `ldd libMagickWand`) via a small scoped symlink
# directory ($COSMOS_DIR/magickwand_libs, symlinked to /host-lib64/<name>),
# plus MAGICK_HOME pointing at a second scoped dir
# ($COSMOS_DIR/magickwand_home/lib) with unversioned-name symlinks
# (libMagickWand-6.Q16.so -> libMagickWand-6.Q16.so.7) so wand's
# `library_paths()` finds it directly via ctypes.CDLL instead of relying on
# `ctypes.util.find_library` (which only consults ldconfig's cache and
# never sees LD_LIBRARY_PATH additions). Verified working end-to-end:
# `from libero.libero.envs import env_wrapper` succeeds with exit code 0.
#
# Do NOT set LD_LIBRARY_PATH via `apptainer exec --env` — that applies to
# the entrypoint bash process too (not just python), which can break bash
# itself (see incident in job 42146973). Set it only inside the bash -c
# string, after bash itself is already running.
apptainer exec --nv \
    --env MUJOCO_GL=egl \
    --env HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache \
    --env TRANSFORMERS_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache \
    --env HF_TOKEN=$(cat /jet/home/asingal/.cache/huggingface/token 2>/dev/null || echo "") \
    --env LIBERO_CONFIG_PATH=$WORKTREE/LIBERO-Safety/.libero_config \
    --env MAGICK_HOME=$COSMOS_DIR/magickwand_home \
    --bind $REPO:$REPO \
    --bind /ocean/projects/cis250185p/asingal/.hf_cache:/ocean/projects/cis250185p/asingal/.hf_cache \
    --bind /jet/home/asingal:/jet/home/asingal \
    --bind /usr/lib64:/host-lib64:ro \
    $SIF_PATH \
    bash -c "cd $WORKTREE && export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$COSMOS_DIR/magickwand_libs\" && $COSMOS_DIR/.venv_rhel8/bin/python \
        $WORKTREE/run_libsafety_cosmos_policy_eval.py \
            --task_suite_name human_safety \
            --task_index 0 \
            --num_trials_per_task 1 \
            --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
            --results_output_dir $WORKTREE/cosmos_benchmark_libsafety \
            --video_output_dir $WORKTREE/cosmos_video_libsafety \
            --local_log_dir $WORKTREE/cosmos_benchmark_libsafety/logs \
            --seed 195 \
            --deterministic True \
            --run_id_note smoke"
