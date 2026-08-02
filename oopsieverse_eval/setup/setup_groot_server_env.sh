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

# FIFTH root-cause fix (see groot_risk_gate_report.md -- this one is a real
# runtime ABI mismatch, not another install-bootstrap ordering bug): the
# prebuilt flash-attn wheel `uv sync` pulls in (built for cp310/linux_x86_64)
# was compiled against glibc >= 2.32; Bridges2 GPU-shared compute nodes run
# an older system glibc (2.28), so `import flash_attn` raises
# `ImportError: /lib64/libc.so.6: version 'GLIBC_2.32' not found` at RUNTIME
# on the actual GPU node (confirmed directly: `python -c "import flash_attn"`
# reproduces this exact error in this venv). No amount of install-ordering
# fixing helps here -- the compiled .so itself is binary-incompatible with
# this cluster. Building from source instead was considered and rejected
# (heavy CUDA-toolchain compile, likely to hit its own issues, too large a
# lift for a risk-gate spike); a container was also rejected as overkill.
#
# Fix: uninstall flash-attn entirely. `transformers.utils.is_flash_attn_2_available()`
# only checks package *presence* (`importlib.util.find_spec`) plus a CUDA
# check -- it does NOT smoke-test-import the compiled extension -- so with
# flash-attn present but broken, transformers' own guarded
# `if is_flash_attn_2_available(): from flash_attn... import ...` in
# modeling_flash_attention_utils.py (triggered merely by importing
# transformers.modeling_utils, i.e. by ANY HF model use, independent of
# per-model attn_implementation choice) evaluates True and crashes. With
# flash-attn uninstalled, that check correctly returns False and the import
# is skipped entirely. VERIFIED directly (not assumed) in this venv:
# `is_flash_attn_2_available()` -> False, `'flash_attn' not in sys.modules`
# holds through `import transformers.modeling_utils` and
# `from gr00t.model.modules.eagle_backbone import EagleBackbone`.
#
# This alone is NOT sufficient, though: GR00T's own bundled
# Eagle-Block2A-2B-v2 modeling code (gr00t/model/modules/nvidia/
# Eagle-Block2A-2B-v2/modeling_eagle3_vl.py) hard-codes
# `config.vision_config._attn_implementation = "flash_attention_2"` and
# `assert config.text_config._attn_implementation == "flash_attention_2"`
# UNCONDITIONALLY in its __init__ (bypassing the availability check
# entirely) -- so even with flash-attn uninstalled, constructing the model
# would still explicitly request flash_attention_2 and fail a proper
# "flash_attn not available" check inside transformers instead. That file
# has been hand-patched in this CLONE_DIR (search "OOPSIEVERSE PATCH") to
# force "eager" instead -- pure-PyTorch, no compiled extension, adequate
# for this non-degeneracy risk-gate spike (slower, not incorrect). This
# patch lives only in the local clone's working tree (not upstream, not
# committed to Isaac-GR00T's own git history) -- if CLONE_DIR is ever
# deleted and re-cloned, gr00t_n1d6_eagle_patch.py below re-applies it.
"$ENV_PREFIX/bin/python" -m pip uninstall -y flash-attn

# Companion patch: the checkpoint's own saved Eagle-Block2A-2B-v2 config sets
# the TOP-LEVEL config._attn_implementation to "flash_attention_2", which
# transformers validates (and raises on) in `_autoset_attn_implementation`
# BEFORE `Eagle3_VLForConditionalGeneration.__init__` (and its own patched
# sub-config handling below) even runs -- so this needs forcing too, one
# level up, in EagleBackbone itself.
EAGLE_BACKBONE_FILE="$CLONE_DIR/gr00t/model/modules/eagle_backbone.py"
if ! grep -q "OOPSIEVERSE PATCH" "$EAGLE_BACKBONE_FILE"; then
    echo "Applying OOPSIEVERSE PATCH (force eager attention, top level) to $EAGLE_BACKBONE_FILE ..."
    python3 - "$EAGLE_BACKBONE_FILE" <<'PYEOF'
import sys

path = sys.argv[1]
text = open(path).read()
old = (
    '            config = AutoConfig.from_pretrained(eagle_path, trust_remote_code=True)\n'
    '            self.model = AutoModel.from_config(config, trust_remote_code=True)\n'
)
new = (
    '            config = AutoConfig.from_pretrained(eagle_path, trust_remote_code=True)\n'
    '            config._attn_implementation = "eager"  # OOPSIEVERSE PATCH\n'
    '            self.model = AutoModel.from_config(config, trust_remote_code=True, attn_implementation="eager")\n'
)
assert old in text, "expected source snippet not found -- upstream file may have changed"
open(path, "w").write(text.replace(old, new))
print("Patched.")
PYEOF
    grep -q "OOPSIEVERSE PATCH" "$EAGLE_BACKBONE_FILE" || {
        echo "ERROR: OOPSIEVERSE PATCH did not apply to $EAGLE_BACKBONE_FILE -- check manually."
        exit 1
    }
else
    echo "OOPSIEVERSE PATCH already present in $EAGLE_BACKBONE_FILE -- skipping."
fi

EAGLE_MODELING_FILE="$CLONE_DIR/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/modeling_eagle3_vl.py"
if ! grep -q "OOPSIEVERSE PATCH" "$EAGLE_MODELING_FILE"; then
    echo "Applying OOPSIEVERSE PATCH (force eager attention) to $EAGLE_MODELING_FILE ..."
    python3 - "$EAGLE_MODELING_FILE" <<'PYEOF'
import re
import sys

path = sys.argv[1]
text = open(path).read()

text = text.replace(
    '            elif config.vision_config.model_type == "siglip_vision_model":\n'
    '                config.vision_config._attn_implementation = "flash_attention_2"\n',
    '            elif config.vision_config.model_type == "siglip_vision_model":\n'
    '                # OOPSIEVERSE PATCH: forced eager, see setup_groot_server_env.sh\n'
    '                config.vision_config._attn_implementation = "eager"\n',
)
text = text.replace(
    '            elif config.vision_config.model_type == "siglip2_vision_model":\n'
    '                config.vision_config._attn_implementation = "flash_attention_2"\n',
    '            elif config.vision_config.model_type == "siglip2_vision_model":\n'
    '                config.vision_config._attn_implementation = "eager"  # OOPSIEVERSE PATCH\n',
)
text = re.sub(
    r'assert \(\s*config\.text_config\._attn_implementation == "flash_attention_2"\s*\), f"(Qwen[23]) must use flash_attention_2 but got \{config\.text_config\._attn_implementation\}"',
    r'config.text_config._attn_implementation = "eager"  # OOPSIEVERSE PATCH',
    text,
)
open(path, "w").write(text)
print("Patched.")
PYEOF
    grep -q "OOPSIEVERSE PATCH" "$EAGLE_MODELING_FILE" || {
        echo "ERROR: OOPSIEVERSE PATCH did not apply (source lines may have changed upstream) -- check $EAGLE_MODELING_FILE manually."
        exit 1
    }
else
    echo "OOPSIEVERSE PATCH already present in $EAGLE_MODELING_FILE -- skipping."
fi

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
