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
python install.py --new_env --behavior1k

# NOTE: install.py's install_behavior1k() clones
# externals/behavior1k (UT-Austin-RobIn/BEHAVIOR-1K@proj/safemanibench) if
# missing, then runs `bash setup.sh --omnigibson --bddl --dataset` inside
# the oopsieverse_b1k env (installs Isaac Sim + OmniGibson + BDDL + the
# BEHAVIOR-1K dataset), then pip installs telemoma/pyspacemouse/mediapipe
# for teleop. This is a large, slow, network-heavy install -- run it on a
# login node (needs outbound internet), not inside a SLURM job.
echo "=== setup_b1k_env.sh done. Verify with: conda run -n oopsieverse_b1k python -c 'import omnigibson' ==="
