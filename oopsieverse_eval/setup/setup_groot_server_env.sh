#!/usr/bin/env bash
# Creates a fresh, isolated `oopsieverse_groot_srv` conda env for serving
# NVIDIA's off-the-shelf `nvidia/GR00T-N1.6-3B` checkpoint via the
# `Isaac-GR00T` package's own inference/eval tooling (see
# ../servers/groot_server.py, ../eval/groot_risk_gate_spike.py).
#
# This is a brand-new env, independent of every other conda env in this
# repo. GR00T's own repo is `pip install -e .`'d from a local (throwaway)
# clone -- NOT vendored into this repo -- so its own pinned deps (a specific
# transformers, the "nvidia/Eagle-Block2A-2B-v2" backbone's custom modeling
# code via trust_remote_code, diffusers, flash-attn, pyzmq for the
# ExternalRobotInferenceClient/PolicyServer wire protocol) are isolated here.
set -uo pipefail

ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/oopsieverse_groot_srv"
CLONE_DIR="/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/Isaac-GR00T"
export HF_HOME="/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/hf"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -p "${ENV_PREFIX}" python=3.10 -y
conda activate "${ENV_PREFIX}"

pip install --upgrade pip setuptools wheel

# Clone Isaac-GR00T read-only into the shared cache dir (not vendored into
# this git repo -- matches this repo's convention of keeping large external
# checkouts out of git, see fastwam_server.py's docstring on fast-wam/FastWAM).
mkdir -p "$(dirname "$CLONE_DIR")"
if [ ! -d "$CLONE_DIR" ]; then
    git clone https://github.com/NVIDIA/Isaac-GR00T.git "$CLONE_DIR"
fi

# CRITICAL (risk-gate research finding, see groot_risk_gate_report.md):
# the repo's `main` branch has already moved on to N1.7 -- its
# gr00t/model/ tree only ships `gr00t_n1d7/`, not `gr00t_n1d6/`, and its
# EmbodimentTag enum recategorizes ROBOCASA_PANDA_OMRON as
# "finetuning-only" (no pretrained action head) rather than "pretrain".
# The nvidia/GR00T-N1.6-3B checkpoint (config.json: architectures=
# ["Gr00tN1d6"]) needs the matching `n1.6.1-release` git tag, which DOES
# ship `gr00t_n1d6/` and tags ROBOCASA_PANDA_OMRON as a genuine "Pretrain
# embodiment tag" (confirmed by reading both files at this tag directly).
# Installing from `main` here would load the wrong model class / possibly
# even fail to find the checkpoint's embodiment tag at all.
git -C "$CLONE_DIR" fetch --tags
git -C "$CLONE_DIR" checkout n1.6.1-release
pip install -e "$CLONE_DIR"

# pyzmq for the ExternalRobotInferenceClient/PolicyServer wire protocol
# (groot_server.py) -- NVIDIA's own recommended serving pattern, distinct
# from every other policy server in this repo (REST/pickle-socket).
pip install pyzmq

# flash-attn is best-effort, same judgment call as openvla_server.py: if the
# build fails, fall back to eager/sdpa attention in groot_server.py rather
# than failing setup outright.
pip install "flash-attn" --no-build-isolation || \
  echo "WARNING: flash-attn build failed -- groot_server.py will fall back to sdpa/eager attention."

mkdir -p "$HF_HOME"

python -c "
import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
import gr00t; print('gr00t package OK:', gr00t.__file__)
from gr00t.data.embodiment_tags import EmbodimentTag
print('ROBOCASA_PANDA_OMRON tag present:', hasattr(EmbodimentTag, 'ROBOCASA_PANDA_OMRON'))
import zmq; print('pyzmq OK:', zmq.zmq_version())
"

echo "Downloading nvidia/GR00T-N1.6-3B into HF_HOME=$HF_HOME ..."
python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('nvidia/GR00T-N1.6-3B')
print('Checkpoint cached at:', path)
"

echo "oopsieverse_groot_srv env ready. Remember to set HF_HOME=$HF_HOME when serving."
