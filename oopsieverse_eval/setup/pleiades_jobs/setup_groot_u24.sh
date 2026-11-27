#!/bin/bash
# Unpatched NVIDIA GR00T N1.6 server env (Isaac-GR00T n1.6.1-release, `uv sync`, flash-attn kept),
# built and run inside the Ubuntu 24.04 apptainer sandbox (glibc 2.39) so the prebuilt flash-attn loads.
set -uo pipefail
export PATH=/data/asingal2/envs/apptainer/bin:$PATH
SB=/data/asingal2/oopsieverse/b1k_sandbox
G=/data/asingal2/oopsieverse/externals/Isaac-GR00T-nv
[ -d $G/.git ] || { git clone -q https://github.com/NVIDIA/Isaac-GR00T.git $G; git -C $G checkout -q n1.6.1-release; }
git -C $G log --oneline -1
git -C $G status --short | head
apptainer exec --nv --bind /data --bind /home/asingal2 $SB bash -c "
  export PATH=/home/asingal2/.local/bin:/data/asingal2/envs/cuda126/bin:\$PATH CUDA_HOME=/data/asingal2/envs/cuda126
  export UV_CACHE_DIR=/data/asingal2/uv_cache UV_PROJECT_ENVIRONMENT=/data/asingal2/envs/groot_srv_u24 DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1
  ldd --version | head -1
  cd $G && uv sync --python 3.10 && /data/asingal2/envs/groot_srv_u24/bin/python -c 'import flash_attn, torch; print(\"flash_attn\", flash_attn.__version__, \"torch\", torch.__version__, torch.cuda.is_available())'
"
echo "U24_EXIT=$?"
