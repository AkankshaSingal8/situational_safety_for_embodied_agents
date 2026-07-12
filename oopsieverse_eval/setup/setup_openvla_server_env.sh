#!/usr/bin/env bash
# Creates a fresh, isolated `oopsieverse_openvla_srv` conda env for serving an
# off-the-shelf `openvla/openvla-7b-finetuned-libero-*` checkpoint over a
# small REST API (see ../servers/openvla_server.py).
#
# This is a brand-new env, independent of any existing OpenVLA env in this
# repo (`openvla_libero_merged`) or the sibling LIBERO-Safety benchmark's
# `libsafety_openvla` env — neither is touched or imported from. The
# checkpoint ships its own modeling code via HF `trust_remote_code`
# (confirmed via its `auto_map` in config.json), so no `openvla/openvla`
# GitHub checkout is needed either — just the pinned generic ML stack below.
set -euo pipefail

ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/oopsieverse_openvla_srv"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -p "${ENV_PREFIX}" python=3.10 -y
conda activate "${ENV_PREFIX}"

pip install --upgrade pip setuptools wheel

# Pinned per OpenVLA's own requirements-min.txt (github.com/openvla/openvla) --
# these versions matter, the project's own docs report regressions on newer
# transformers/timm releases.
pip install \
  torch==2.2.0 \
  torchvision==0.17.0 \
  torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu118

pip install \
  "numpy<2" \
  transformers==4.40.1 \
  tokenizers==0.19.1 \
  timm==0.9.10 \
  accelerate>=0.25.0 \
  peft==0.11.1 \
  pillow \
  sentencepiece==0.1.99 \
  protobuf \
  rich \
  json-numpy \
  fastapi \
  uvicorn[standard]

# flash-attn is best-effort: if the build fails (old pin vs newer CUDA/driver
# stacks, a known issue per project research), fall back to
# attn_implementation="sdpa" in openvla_server.py rather than failing setup.
pip install "flash-attn==2.5.5" --no-build-isolation || \
  echo "WARNING: flash-attn build failed -- openvla_server.py will fall back to sdpa attention."

# Route HF downloads to $OCEAN (project scratch), NOT $HOME -- home quota on
# this cluster is nearly full (~1.6GB free as of this build), and the 7B
# checkpoint alone is ~14GB in bf16.
export HF_HOME="/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/hf"
mkdir -p "$HF_HOME"

python -c "
import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
import transformers; print('transformers:', transformers.__version__)
import fastapi, uvicorn, json_numpy; print('server deps: OK')
"
echo "oopsieverse_openvla_srv env ready. Remember to set HF_HOME=$HF_HOME when serving."
