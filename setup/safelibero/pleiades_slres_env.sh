#!/bin/bash
# Build the environments for experiments/safelibero_residual on the CMU pleiades cluster.
#   sbatch -p pleiades --time=03:00:00 -c 8 --mem=32G --gres=gpu:1 \
#          -o /data/asingal2/slurm_logs/env_%j.out setup/safelibero/pleiades_slres_env.sh
# (needs a GPU only for the final import/CUDA checks)
#
# 1. slres   (micromamba, Python 3.10): SafeLIBERO/LIBERO sim, base flow policy, residual flow,
#            BGE text embeddings, openpi websocket client.  Pins follow the PSC env
#            `openvla_libero_merged` that produced the Spatial results (torch 2.2 cu118,
#            robosuite 1.4.1, numpy 1.24.4) plus mujoco 3.2.3 as in the POC.
# 2. openpi  (uv, Python 3.11): pi0.5 policy server from vlsa-aegis/openpi (same code as the
#            reported pi0.5 SafeLIBERO numbers).
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENVS=${ENVS:-/data/asingal2/envs}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/data/asingal2/micromamba}
export PIP_CACHE_DIR=/data/asingal2/pip_cache UV_CACHE_DIR=/data/asingal2/uv_cache
MM=$HOME/.local/bin/micromamba
UV=$HOME/.local/bin/uv
mkdir -p "$ENVS" "$MAMBA_ROOT_PREFIX"

BUILD=${BUILD:-all}   # slres | openpi | all
# ---------------------------------------------------------------- slres
P=$ENVS/slres
PY=$P/bin/python
if [ "$BUILD" != openpi ]; then
if [ ! -x "$P/bin/python" ]; then
  $MM create -y -p "$P" -c conda-forge python=3.10 "libstdcxx-ng>=12" "av>=10" ffmpeg
fi
$PY -m pip install --upgrade "pip<25" "setuptools<70" wheel
$PY -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
$PY -m pip install \
  "numpy==1.24.4" mujoco==3.2.3 robosuite==1.4.1 bddl==1.0.1 gym==0.25.2 \
  hydra-core==1.2.0 easydict==1.9 future==0.18.2 cloudpickle==2.1.0 \
  opencv-python-headless==4.8.1.78 h5py imageio imageio-ffmpeg scipy matplotlib tqdm \
  "transformers==4.40.2" "tokenizers<0.20" "huggingface_hub<0.26" safetensors \
  websockets msgpack pyyaml pytest pyopengl==3.1.7 "termcolor" "einops"
$PY -m pip install -e "$REPO_ROOT/SafeLIBERO/safelibero" --no-deps
if [ -d "$REPO_ROOT/vlsa-aegis/openpi/packages/openpi-client" ]; then
  $PY -m pip install -e "$REPO_ROOT/vlsa-aegis/openpi/packages/openpi-client" --no-deps
  $PY -m pip install "dm-tree" "tree" || true
fi

# LIBERO asks for its config interactively on first import; write it non-interactively.
LCFG=$HOME/.safelibero; mkdir -p "$LCFG"   # SafeLIBERO reads LIBERO_CONFIG_PATH or ~/.safelibero
LROOT=$REPO_ROOT/SafeLIBERO/safelibero/libero/libero
cat > "$LCFG/config.yaml" <<EOF
benchmark_root: $LROOT
bddl_files: $LROOT/bddl_files
init_states: $LROOT/init_files
datasets: /data/asingal2/data_libero
assets: $LROOT/assets
EOF
# Upstream SafeLIBERO ships three scene XMLs with mesh paths hardcoded to the authors' machine
# (/mnt/workspace/openpi/third_party/{safe,}libero/...); point them at this checkout.
for f in libero_tabletop_base_style.xml basket.xml basket_asset.xml; do
  sed -i -E "s#/mnt/workspace/openpi/third_party/(safe)?libero/libero/libero/assets/scenes#$LROOT/assets/scenes#g" "$LROOT/assets/scenes/$f"
done
fi

# ---------------------------------------------------------------- openpi (pi0.5 server)
OP=$REPO_ROOT/vlsa-aegis/openpi
if [ "$BUILD" != slres ] && [ -d "$OP" ]; then
  [ -d "$OP/packages/openpi-client" ] && $PY -m pip install -e "$OP/packages/openpi-client" --no-deps
  export UV_PROJECT_ENVIRONMENT=$ENVS/openpi UV_PYTHON_INSTALL_DIR=/data/asingal2/uv_python
  (cd "$OP" && GIT_LFS_SKIP_SMUDGE=1 $UV sync --python 3.11 --no-install-package rerun-sdk)
  # --no-install-package rerun-sdk: its wheels need glibc 2.31 (RHEL8 has 2.28); visualization only, unused
  # by the server. uv sync already installs openpi itself (editable), so no separate pip install.
fi

# ---------------------------------------------------------------- checks
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
$PY - <<'EOF'
import torch, mujoco, robosuite, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("mujoco", mujoco.__version__, "robosuite", robosuite.__version__, "numpy", numpy.__version__)
from libero.libero import benchmark
bd = benchmark.get_benchmark_dict()
print("safelibero suites:", sorted(k for k in bd if k.startswith("safelibero")))
import transformers; print("transformers", transformers.__version__)
try:
    import openpi_client; print("openpi_client OK")
except Exception as e:
    print("openpi_client missing:", e)
EOF
if [ -x "$ENVS/openpi/bin/python" ]; then
  XLA_PYTHON_CLIENT_PREALLOCATE=false $ENVS/openpi/bin/python -c "import jax, openpi; print('jax', jax.__version__, jax.devices())"
fi
echo ENV_DONE
