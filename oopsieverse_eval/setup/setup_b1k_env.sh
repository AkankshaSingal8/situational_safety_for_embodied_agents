#!/usr/bin/env bash
# Creates the `oopsieverse_b1k` conda env and installs BEHAVIOR-1K/OmniGibson
# (UT-Austin-RobIn/BEHAVIOR-1K@proj/safemanibench) into it, using OopsieVerse's
# own install.py (unmodified). Mirrors setup_robocasa_env.sh's invocation
# pattern but for --behavior1k. Safe to re-run: install.py skips steps that
# already exist (env, clones).
#
# ISOLATION: only creates/touches the `oopsieverse_b1k` conda env (a brand
# new named env, resolved to /ocean/projects/cis250185p/asingal/envs via this
# account's ~/.condarc envs_dirs). Does NOT read from or write into
# oopsieverse_robocasa, oopsieverse_openvla_srv, oopsieverse_pi05_srv,
# oopsieverse_fastwam_srv, or the Cosmos Apptainer container.
#
# HOME-QUOTA SAFETY (read before running): /jet/home/$USER on Bridges2 is a
# small (25G) quota filesystem and was already at 99% full / ~473M free when
# this script was written. Isaac Sim / Omniverse Kit's installer/runtime
# writes shader/extension/download caches to a handful of *fixed*
# non-XDG-configurable paths under $HOME by default
# (~/.cache/ov, ~/.nvidia-omniverse, ~/.local/share/ov,
# ~/.local/share/omniverse) regardless of XDG_CACHE_HOME (which this
# account already redirects to /ocean for everything that respects it).
# To avoid silently filling $HOME mid-install, this script pre-creates
# those directories on /ocean and symlinks them into $HOME *before*
# running BEHAVIOR-1K's setup.sh. This step is skipped (with a warning) if
# a real (non-symlink) directory already exists at any of those paths, so
# it never clobbers pre-existing data from other projects.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OOPSIEVERSE_DIR="$SCRIPT_DIR/../external/oopsieverse"
CACHE_ROOT="/ocean/projects/cis250185p/asingal/.b1k_home_cache_redirect"

redirect_home_dir() {
    local home_path="$1"
    local ocean_path="$2"
    mkdir -p "$ocean_path"
    if [ -L "$home_path" ]; then
        echo "[INFO] $home_path already a symlink, leaving as-is."
    elif [ -e "$home_path" ]; then
        echo "[WARN] $home_path already exists and is NOT a symlink -- leaving" \
             "it in place (not redirecting to /ocean). Watch \$HOME quota" \
             "during install."
    else
        mkdir -p "$(dirname "$home_path")"
        ln -s "$ocean_path" "$home_path"
        echo "[INFO] Redirected $home_path -> $ocean_path"
    fi
}

echo "=== Pre-install: redirecting known Isaac Sim/Omniverse Kit cache dirs off \$HOME ==="
redirect_home_dir "$HOME/.cache/ov" "$CACHE_ROOT/cache_ov"
redirect_home_dir "$HOME/.nvidia-omniverse" "$CACHE_ROOT/nvidia_omniverse"
redirect_home_dir "$HOME/.local/share/ov" "$CACHE_ROOT/local_share_ov"
redirect_home_dir "$HOME/.local/share/omniverse" "$CACHE_ROOT/local_share_omniverse"
df -h "$HOME" | tail -1

cd "$OOPSIEVERSE_DIR"

# Step 1: let install.py create the env + clone
# externals/behavior1k (UT-Austin-RobIn/BEHAVIOR-1K@proj/safemanibench) --
# this is the unmodified install.py invocation. It also then tries to run
# `bash setup.sh --omnigibson --bddl --dataset` itself, but install.py does
# NOT expose a way to pass BEHAVIOR-1K setup.sh's --accept-*-tos/eula flags,
# so that inner call hits an interactive
# "Do you accept ALL of the above terms? (y/N)" prompt and fails with a
# non-zero exit under a non-interactive/background shell (confirmed: this
# is exactly what happened on the first run of this script). We tolerate
# that first failure with `|| true` -- by the time it happens, the env and
# the externals/behavior1k clone already exist -- then run the real install
# step 2 below directly against setup.sh with the accept flags.
echo "=== Step 1/2: python install.py --new_env --behavior1k (env + clone; the bundled setup.sh call inside this is EXPECTED to fail on the interactive EULA prompt -- that's handled in step 2) ==="
python install.py --new_env --behavior1k || true

if [ ! -d externals/behavior1k ]; then
    echo "[ERROR] externals/behavior1k was not cloned by install.py -- aborting."
    exit 1
fi

# Step 2: run BEHAVIOR-1K's own setup.sh directly (not through install.py)
# with the three --accept-*-tos/eula flags so it runs non-interactively.
# This is the SAME command install.py runs internally
# (`bash setup.sh --omnigibson --bddl --dataset`), with only the accept
# flags added -- not a fork/modification of BEHAVIOR-1K's setup.sh itself.
# Flags confirmed by reading externals/behavior1k/setup.sh directly (its
# --help / argument parsing), not guessed:
#   --accept-conda-tos    Automatically accept Conda Terms of Service
#   --accept-nvidia-eula  Automatically accept NVIDIA Isaac Sim EULA
#   --accept-dataset-tos  Automatically accept BEHAVIOR Dataset Terms
# NOTE: setup.sh's own --new-env flag is intentionally NOT passed -- that
# would create a *second*, differently-named ('behavior') conda env inside
# the BEHAVIOR-1K repo, defeating the point of installing into
# oopsieverse_b1k. Without --new-env, setup.sh installs into whatever conda
# env is currently active, which `conda run -n oopsieverse_b1k` sets up.
echo "=== Step 2/2: bash setup.sh --omnigibson --bddl --dataset --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos (inside oopsieverse_b1k) ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda run --no-capture-output -n oopsieverse_b1k \
    bash -c "cd externals/behavior1k && bash setup.sh --omnigibson --bddl --dataset --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos"

# Same as install.py's install_behavior1k() does after a successful
# setup.sh run: install teleop deps.
echo "=== Installing teleop dependencies (telemoma, pyspacemouse, mediapipe) ==="
conda run --no-capture-output -n oopsieverse_b1k \
    pip install telemoma==0.3.0 pyspacemouse==1.1.4 mediapipe==0.10.21

echo "=== setup_b1k_env.sh done. Verify with: conda run -n oopsieverse_b1k python -c 'import omnigibson' ==="
