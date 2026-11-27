#!/bin/bash
# NVIDIA's own GR00T RoboCasa eval client env (Isaac-GR00T n1.6.1-release, gr00t/eval/sim/robocasa/setup_RoboCasa.sh):
# squarefk/robocasa@d89d481 + robosuite@master + kitchen assets. Used only to check our GR00T server
# against NVIDIA's published OpenSingleDoor number (81.5%).
set -euo pipefail
export PATH=/home/asingal2/.local/bin:$PATH UV_CACHE_DIR=/data/asingal2/uv_cache
G=/data/asingal2/oopsieverse/externals/Isaac-GR00T
RC=$G/external_dependencies/robocasa
V=$G/gr00t/eval/sim/robocasa/robocasa_uv
if [ ! -f $RC/setup.py ]; then
  rm -rf $RC; git clone -q https://github.com/squarefk/robocasa $RC; git -C $RC checkout -q d89d481ce9c76da7f179466981676e268aa842e5
fi
git -C $RC log --oneline -1
[ -x $V/.venv/bin/python ] || uv venv $V/.venv --python 3.10
source $V/.venv/bin/activate
uv pip install -q setuptools wheel
uv pip install -q torch==2.5.1 torchvision==0.20.1
uv pip install -q "git+https://github.com/ARISE-Initiative/robosuite.git@master"
uv pip install -q -e $RC --config-settings editable_mode=compat
uv pip install -q gymnasium==0.29.1 pydantic av==15.0.0 zmq transformers==4.51.3 msgpack==1.1.0 msgpack-numpy==0.4.8 tyro
uv pip install -q --editable $G --no-deps
python -c "import robocasa, robosuite; print('robosuite', robosuite.__version__)"
python $RC/robocasa/scripts/setup_macros.py || true
yes y | python $RC/robocasa/scripts/download_kitchen_assets.py
du -sh $RC/robocasa/models/assets 2>/dev/null
echo NV_SETUP_DONE
