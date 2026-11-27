#!/bin/bash
# GR00T-N1.6 BEHAVIOR eval stack (Isaac-GR00T examples/BEHAVIOR/README.md): BEHAVIOR-1K feat/task-progress +
# GR00T uv env + Isaac Sim 4.5 + 2025 challenge task instances. Built and run inside the Ubuntu 24.04 sandbox on
# pleiades-1-10 (Isaac Sim segfaults on the Ada nodes' driver).
set -uo pipefail
export PATH=/data/asingal2/envs/apptainer/bin:$PATH
SB=/data/asingal2/oopsieverse/b1k_sandbox
B=/data/asingal2/oopsieverse/externals/behavior1k_tp
GNV=/data/asingal2/oopsieverse/externals/Isaac-GR00T-nv
E=/data/asingal2/envs/groot_b1k
if [ ! -f $B/setup_uv.sh ]; then
  rm -rf $B
  git clone -q --filter=blob:none --no-checkout -b feat/task-progress https://github.com/StanfordVL/BEHAVIOR-1K.git $B
  git -C $B sparse-checkout init --no-cone
  printf '/*\n!/joylo/\n!/docs/\n/docs/assets/OmniGibson_logo.png\n' > $B/.git/info/sparse-checkout
  git -C $B checkout -q feat/task-progress
fi
git -C $B log --oneline -1
mkdir -p $B/docs/assets; [ -s $B/docs/assets/OmniGibson_logo.png ] || curl -sL -o $B/docs/assets/OmniGibson_logo.png \
  https://raw.githubusercontent.com/StanfordVL/BEHAVIOR-1K/feat/task-progress/docs/assets/OmniGibson_logo.png
apptainer exec --nv --bind /data --bind /home/asingal2 $SB bash -c "
  set -uo pipefail
  export PATH=/home/asingal2/.local/bin:/data/asingal2/envs/cuda126/bin:\$PATH CUDA_HOME=/data/asingal2/envs/cuda126
  export UV_CACHE_DIR=/data/asingal2/uv_cache UV_PROJECT_ENVIRONMENT=$E DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1
  export OMNIGIBSON_DATA_PATH=/data/asingal2/oopsieverse/b1k_tp_data OMNI_KIT_ACCEPT_EULA=YES
  mkdir -p \$OMNIGIBSON_DATA_PATH
  cd $GNV && uv sync --python 3.10 && uv pip install --python $E/bin/python -e . || exit 11
  source $E/bin/activate
  cd $B && bash ./setup_uv.sh || exit 12
  uv pip install --python $E/bin/python 'pydantic<2.10' || true
  cd $GNV && python gr00t/eval/sim/BEHAVIOR/prepare_test_instances.py || exit 13
  python -c 'import omnigibson, bddl; from omnigibson.learning.wrappers import TaskProgressWrapper; print(\"OG_TP_OK\")'
"
echo "SETUP_EXIT=$?"
