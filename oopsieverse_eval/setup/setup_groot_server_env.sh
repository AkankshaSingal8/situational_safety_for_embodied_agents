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

# CRITICAL ordering fix (first attempt at this failed -- see
# groot_risk_gate_report.md "install ordering" postmortem): this tag's
# pyproject.toml declares BOTH torch==2.7.1 AND flash-attn==2.7.4.post1 as
# hard, non-optional base dependencies (no extras group to skip flash-attn).
# flash-attn's own setup.py does `import torch` during pip's isolated
# build-metadata step -- if torch isn't already present in the target env
# BEFORE this install runs, pip's per-package build isolation gives
# flash-attn's setup.py a fresh venv with no torch in it, so the build
# fails with `ModuleNotFoundError: No module named 'torch'`. Worse, because
# `pip install -e .` resolves/builds all of this tag's dependencies in one
# atomic command, that single flash-attn failure aborted the ENTIRE
# install -- so numpy/huggingface_hub/transformers etc. (all otherwise-fine
# deps bundled in the same command) never got installed either, and every
# downstream check (including the spike script's own `import numpy`)
# failed on the very first line.
#
# Fix: install torch FIRST as its own step (so it's importable in the
# current env), then install Isaac-GR00T with `--no-build-isolation` (pip
# builds every package in the command against the current env instead of a
# fresh isolated one per package, so flash-attn's setup.py now finds torch).
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print('torch preinstalled OK:', torch.__version__)"

# SECOND install-ordering fix (job 42907882 failed on this next line, same
# failure *class* as the flash-attn one above -- a transitive build
# dependency needing something this shell doesn't provide by default):
# gr00t==0.1.0 also hard-depends on deepspeed==0.17.6, whose setup.py
# unconditionally probes `CUDA_HOME` at *metadata-generation* time (before
# any actual compilation) and raises `MissingCUDAException` if it's unset --
# true here because this script never loaded a CUDA toolkit module, only
# relying on torch's bundled cu128 runtime libs (which don't set CUDA_HOME
# or provide nvcc). Because `pip install -e .` is one atomic command, that
# single deepspeed failure again aborted the whole install, so numpy/
# huggingface_hub/gr00t itself never landed -- identical cascade shape to
# the flash-attn bug, different package.
#
# Fix: `module load cuda/12.6.1` (closest available toolkit to torch's
# cu128 build; confirmed on this cluster to set CUDA_HOME=/opt/packages/
# cuda/v12.6.1 and put nvcc on PATH) BEFORE this install, so deepspeed's
# (and flash-attn's) CUDA op builders find a real toolkit. DS_BUILD_OPS=0 /
# DS_SKIP_CUDA_CHECK=1 are set as a second line of defense in case the
# module's CUDA_HOME still isn't picked up in some shell -- we only need
# GR00T for single-GPU zero-shot inference here, not deepspeed's
# distributed-training custom ops, so skipping deepspeed's AOT op
# compilation (it falls back to JIT-on-first-use, never triggered by
# inference-only code) is safe for this spike's purposes.
module load cuda/12.6.1
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1

# THIRD install-ordering fix (job 42909156 failed on this same line again,
# third distinct root cause under the identical failure shape -- confirmed
# by reading flash-attn==2.7.4.post1's setup.py directly at that exact
# pinned tag rather than guessing): flash-attn's `NinjaBuildExtension
# .__init__()` does `import psutil` unconditionally, and setuptools
# eagerly instantiates that class during plain metadata/sdist generation
# (via `_add_defaults_ext()` -> `get_finalized_command('build_ext')`) --
# i.e. `psutil` is needed even just to compute flash-attn's package
# metadata, not only to actually compile it. It wasn't present because
# nothing in this script had installed it yet.
#
# Fix: install psutil (+ ninja, ninja being flash-attn's declared
# `setup_requires` build tool) as their own explicit, verified step BEFORE
# the big gr00t install -- per the "smaller, verified steps instead of one
# atomic command" principle: each of these three bootstrap-dependency bugs
# (torch, CUDA_HOME, psutil) was hidden behind one all-or-nothing `pip
# install -e .` swallowing the real error under a generic
# metadata-generation-failed banner, so from here on any NEW failure in
# this same command is presumed to be a genuine, different root cause,
# not a fourth variant of "one more transitive build dep missing" --
# check the actual traceback line-by-line before further edits, per this
# postmortem's own advice to future runs.
pip install psutil ninja
python -c "import psutil, ninja; print('psutil/ninja OK:', psutil.__version__, ninja.__version__)"

pip install -e "$CLONE_DIR" --no-build-isolation

# pyzmq for the PolicyServer/PolicyClient ZMQ REQ/REP wire protocol
# (groot_server.py) -- NVIDIA's own recommended serving pattern, distinct
# from every other policy server in this repo (REST/pickle-socket).
pip install pyzmq

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
