#!/usr/bin/env bash
# Installs BEHAVIOR-1K/OmniGibson/Isaac Sim into the `oopsieverse_b1k` conda
# env, run INSIDE the b1k_container.sif Apptainer container (Ubuntu 24.04,
# glibc 2.39). This is the container-side counterpart to setup_b1k_env.sh --
# call this script via `apptainer exec --nv b1k_container.sif bash
# setup_b1k_env_container.sh`, not directly on the host (bare-metal RHEL 8.10
# has glibc 2.28, too old for Isaac Sim 4.5.0's pip wheel -- see
# task-1-report.md for the exact error that motivated this container).
#
# Uses the container's OWN conda (/opt/miniconda3, baked into the .sif by
# b1k_container.def), not the host's miniconda -- a conda binary linked
# against the host's older glibc previously needed a from-scratch rebuild
# once cosmos-policy hit the same class of problem (see
# build_cosmos_sif.slurm's "rebuild venv inside container" step); we do the
# analogous thing here for conda envs from the start rather than reusing the
# broken bare-metal attempt.
#
# $HOME (and therefore ~/.condarc, which already points envs_dirs/pkgs_dirs
# at /ocean) is bind-mounted into the container by Apptainer by default, so
# the container's conda still resolves envs/pkgs to /ocean, not into the
# container's small overlay.
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
        echo "[WARN] $home_path already exists and is NOT a symlink -- leaving it in place."
    else
        mkdir -p "$(dirname "$home_path")"
        ln -s "$ocean_path" "$home_path"
        echo "[INFO] Redirected $home_path -> $ocean_path"
    fi
}

echo "=== [container] glibc/OS check ==="
ldd --version | head -1
cat /etc/os-release | head -2

echo "=== [container] Pre-install: redirecting Isaac Sim/Omniverse Kit cache dirs off \$HOME ==="
redirect_home_dir "$HOME/.cache/ov" "$CACHE_ROOT/cache_ov"
redirect_home_dir "$HOME/.nvidia-omniverse" "$CACHE_ROOT/nvidia_omniverse"
redirect_home_dir "$HOME/.local/share/ov" "$CACHE_ROOT/local_share_ov"
redirect_home_dir "$HOME/.local/share/omniverse" "$CACHE_ROOT/local_share_omniverse"
df -h "$HOME" | tail -1

# Use the container's own conda explicitly (PATH already prefers it per
# b1k_container.def's %environment, but be explicit/robust here).
CONDA_BIN=/opt/miniconda3/bin/conda
source /opt/miniconda3/etc/profile.d/conda.sh

cd "$OOPSIEVERSE_DIR"

echo "=== Step 1/2: $CONDA_BIN create -y -n oopsieverse_b1k python=3.10 (container conda) ==="
if ! conda env list | grep -q "^oopsieverse_b1k "; then
    "$CONDA_BIN" create -y -n oopsieverse_b1k python=3.10
else
    echo "[INFO] oopsieverse_b1k env already exists (this container-built one) -- reusing."
fi

if [ ! -d externals/behavior1k ]; then
    echo "[INFO] Cloning BEHAVIOR-1K (proj/safemanibench)..."
    git clone --branch proj/safemanibench https://github.com/UT-Austin-RobIn/BEHAVIOR-1K.git externals/behavior1k
else
    echo "[INFO] externals/behavior1k already present, reusing clone."
fi

echo "=== Step 2/2: bash setup.sh --omnigibson --bddl --dataset --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos (inside container, oopsieverse_b1k env) ==="
conda activate oopsieverse_b1k
cd externals/behavior1k
bash setup.sh --omnigibson --bddl --dataset --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos

echo "=== Installing teleop dependencies (telemoma, pyspacemouse, mediapipe) ==="
pip install telemoma==0.3.0 pyspacemouse==1.1.4 mediapipe==0.10.21

echo "=== setup_b1k_env_container.sh done. Verify with:"
echo "    apptainer exec --nv b1k_container.sif bash -c 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate oopsieverse_b1k && python -c \"import omnigibson\"'"
