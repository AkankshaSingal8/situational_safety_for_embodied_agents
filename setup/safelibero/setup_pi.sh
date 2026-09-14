
# Repo root, derived rather than hardcoded. Override with REPO_ROOT.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd vlsa-aegis
pip install -r requirements.txt
pip install av --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu118
cd main
#change llvmlite==0.36.0 to llvmlite>=0.40.0 in main/requirements.txt 
pip install -r requirements.txt
# 1. Fix llvmlite first
pip install llvmlite==0.40.1

# 2. Install requirements without breaking env
pip install -r requirements.txt --no-deps

# 3. Install missing core libs manually
pip install numba imageio opencv-python cvxpy

pip install \
    tqdm tyro scipy imageio-ffmpeg \
    opencv-python h5py huggingface-hub \
    flask dash ipython ipywidgets

pip install typeguard==2.13.3
# incase it errors, try:
pip uninstall tensorflow-addons -y
pip install typeguard==4.5.1

export MUJOCO_GL=egl
export PYTHONPATH=$PYTHONPATH:"$REPO_ROOT"/vlsa-aegis

pip install gsutil
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero ./checkpoints/

cd "$REPO_ROOT"/vlsa-aegis
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
pip install -e . --no-build-isolation

pip install flax