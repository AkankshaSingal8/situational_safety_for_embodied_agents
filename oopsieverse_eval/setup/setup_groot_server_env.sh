#!/usr/bin/env bash
# Creates a fresh, isolated `oopsieverse_groot_srv` env for serving NVIDIA's
# off-the-shelf `nvidia/GR00T-N1.6-3B` checkpoint via the `Isaac-GR00T`
# package's own inference/eval tooling (see ../servers/groot_server.py,
# ../eval/groot_risk_gate_spike.py).
#
# ARCHITECTURE NOTE (4th and final root-cause fix in this env-setup
# postmortem -- see groot_risk_gate_report.md for the full trail): the
# first three attempts used plain conda + `pip install -e .`, and hit THREE
# separate one-off bootstrap-dependency failures in a row (torch missing
# for flash-attn's build, CUDA_HOME missing for deepspeed's build, psutil
# missing for flash-attn's metadata step) before hitting a FOURTH,
# structurally different failure: `pip._vendor...BackendUnavailable:
# Cannot import 'wheel_stub.buildapi'` while resolving `tensorrt`.
#
# Reading this tag's pyproject.toml directly (not guessed) explains why:
# it has a `[tool.uv.sources]` block pointing `flash-attn` at a PREBUILT
# wheel URL for exactly cp310/linux_x86_64 (no source build at all under
# uv), and a `[[tool.uv.index]]` block pointing `tensorrt` at NVIDIA's own
# `pypi.nvidia.com` index (a real prebuilt wheel there, not the misleading
# `wheel_stub` placeholder sdist that plain pip pulls from default PyPI).
# Neither of these `[tool.uv.*]` sections mean anything to plain pip --
# only `uv` understands them. This package is designed to be installed
# with `uv`, not raw pip (NVIDIA's own getting-started docs already used
# `uv run python gr00t/eval/run_gr00t_server.py ...`, which this script's
# first drafts didn't take literally enough). This tag also ships a
# committed `uv.lock`, i.e. an exact, NVIDIA-verified resolution --
# `uv sync` uses it directly instead of re-resolving from scratch.
#
# Switching to `uv sync` eliminates the whole CLASS of one-off pip
# build-bootstrap bugs above in one shot, rather than patching them
# one-at-a-time forever. deepspeed still builds from source under uv (no
# wheel override for it in pyproject.toml), so the CUDA_HOME fix from the
# second postmortem entry is kept.
set -uo pipefail

ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/oopsieverse_groot_srv"
CLONE_DIR="/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/Isaac-GR00T"
export HF_HOME="/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/hf"

# `module load cuda/12.6.1` -- closest available toolkit to torch's cu128
# build; confirmed on this cluster to set CUDA_HOME=/opt/packages/cuda/
# v12.6.1 and put nvcc on PATH -- deepspeed's setup.py probes CUDA_HOME at
# metadata-generation time and raises MissingCUDAException if unset.
# DS_BUILD_OPS=0 / DS_SKIP_CUDA_CHECK=1 as a second line of defense -- we
# only need GR00T for single-GPU zero-shot inference, not deepspeed's
# distributed-training custom ops, so skipping AOT op compilation
# (falls back to JIT-on-first-use, never triggered by inference-only code)
# is safe for this spike's purposes.
module load cuda/12.6.1
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1

# `uv` is already available on this cluster (confirmed: /jet/home/asingal/
# miniconda3/bin/uv, v0.11.6) -- no separate install step needed. Fall back
# to a user pip install if that ever changes.
command -v uv >/dev/null 2>&1 || pip install --user uv

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
# ship `gr00t_n1d6/`, tags ROBOCASA_PANDA_OMRON as a genuine "Pretrain
# embodiment tag", AND ships the committed `uv.lock` this script relies on.
git -C "$CLONE_DIR" fetch --tags
git -C "$CLONE_DIR" checkout n1.6.1-release

# `uv sync` reads pyproject.toml + uv.lock and creates/populates a venv at
# UV_PROJECT_ENVIRONMENT (redirected here to this repo's usual
# /ocean/.../envs/<name> path instead of uv's default `.venv` under
# CLONE_DIR, to match every other setup_*_server_env.sh in this repo).
export UV_PROJECT_ENVIRONMENT="$ENV_PREFIX"
( cd "$CLONE_DIR" && uv sync --python 3.10 )

# pyzmq is already a direct dependency in gr00t's own pyproject.toml
# (pyzmq==27.0.1) -- installed by `uv sync` above, no separate step needed.

mkdir -p "$HF_HOME"

"$ENV_PREFIX/bin/python" -c "
import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
import gr00t; print('gr00t package OK:', gr00t.__file__)
from gr00t.data.embodiment_tags import EmbodimentTag
print('ROBOCASA_PANDA_OMRON tag present:', hasattr(EmbodimentTag, 'ROBOCASA_PANDA_OMRON'))
import zmq; print('pyzmq OK:', zmq.zmq_version())
"

echo "Downloading nvidia/GR00T-N1.6-3B into HF_HOME=$HF_HOME ..."
"$ENV_PREFIX/bin/python" -c "
from huggingface_hub import snapshot_download
path = snapshot_download('nvidia/GR00T-N1.6-3B')
print('Checkpoint cached at:', path)
"

echo "oopsieverse_groot_srv env ready at $ENV_PREFIX (uv-managed venv, not conda)."
echo "Remember to set HF_HOME=$HF_HOME and activate via 'source $ENV_PREFIX/bin/activate' when serving."
