# LIBERO-Safety Benchmark Replication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up an isolated clone of the LIBERO-Safety benchmark (arXiv:2606.23686) in this repo and get eval-only smoke runs working for pi0, pi0.5, OpenVLA, and OpenVLA-OFT, using the same setup-script + eval-driver conventions already used for this repo's SafeLIBERO evals — without touching any existing branch, env, or repo directory.

**Architecture:** New git branch + new submodule (`LIBERO-Safety/`) sibling to `SafeLIBERO/`. Three new conda envs mirroring the existing three-way split already used for SafeLIBERO (`aegis` → `libsafety_openpi`; `libero_env` → `libsafety_client`; `openvla_libero_merged` → `libsafety_openvla`). Two new eval-driver scripts at repo root (`run_libsafety_eval_openvla.py`, `run_libsafety_eval_openpi.py`) mirroring `run_libero_eval.py` and `vlsa-aegis/main/pi05_evaluation.py`.

**Tech Stack:** conda, pip, JAX/openpi (pi0, pi0.5), PyTorch/transformers/openvla-oft (OpenVLA, OpenVLA-OFT), robosuite 1.4/LIBERO, HuggingFace Hub, gsutil.

## Global Constraints

- Do not modify, `git add`, or `git commit` anything inside `SafeLIBERO/`, `openvla-oft/`, `vlsa-aegis/`, or the existing conda envs (`aegis`, `libero_env`, `openvla_libero_merged`, `qwen`, etc.).
- All new conda envs are created with `conda create -p <ENV_PREFIX>` under `/ocean/projects/cis250185p/asingal/envs/`, never `-n` into the shared named-env namespace used by unrelated tools.
- All new work happens on branch `libero-safety-benchmark`, branched from `full_working_pipeline` at commit `6d243a8` (current HEAD after the spec-doc commit).
- Use off-the-shelf standard LIBERO-pretrained checkpoints only — never the paper's own fine-tuned checkpoints (`LIBERO-Safety/pi05_libero_safety` etc.).
- `export MUJOCO_GL=egl` is required for any headless rendering step on Bridges2 (per this repo's CLAUDE.md).
- Every new setup script and eval script lives at the project root, following the existing naming pattern (`*_setup.sh`, `run_*_eval*.py`), not nested inside `LIBERO-Safety/`.

---

### Task 1: Branch + submodule the LIBERO-Safety repo

**Files:**
- Modify: `.gitmodules`
- Create (submodule): `LIBERO-Safety/`

**Interfaces:**
- Produces: a cloned, uninitialized-assets copy of `github.com/LIBERO-SAFETY/LIBERO-Safety` at `LIBERO-Safety/`, on branch `libero-safety-benchmark`.

- [ ] **Step 1: Create and switch to the new branch**

```bash
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
git checkout full_working_pipeline
git pull origin full_working_pipeline
git checkout -b libero-safety-benchmark
```

Expected: `Switched to a new branch 'libero-safety-benchmark'`

- [ ] **Step 2: Add LIBERO-Safety as a git submodule**

```bash
git submodule add https://github.com/LIBERO-SAFETY/LIBERO-Safety.git LIBERO-Safety
```

Expected: clones into `LIBERO-Safety/`, adds an entry to `.gitmodules`:
```
[submodule "LIBERO-Safety"]
	path = LIBERO-Safety
	url = https://github.com/LIBERO-SAFETY/LIBERO-Safety.git
```

- [ ] **Step 3: Verify the clone and enumerate the repo's task-suite registry**

```bash
ls LIBERO-Safety
find LIBERO-Safety/libero/libero/benchmark -maxdepth 1 -iname "*.py"
```

Expected: `benchmark_scripts/`, `libero/`, `requirements.txt`, `extra_requirements.txt`, `setup.py`, `third_party/robosuite-1.4/` are present. Note (don't guess) the actual benchmark registry file names printed — Task 7/8 will import from here to discover real task-suite name strings (do not hardcode suite names from the SafeLIBERO repo; this repo's are almost certainly different).

- [ ] **Step 4: Commit**

```bash
git add .gitmodules LIBERO-Safety
git commit -m "$(cat <<'EOF'
chore: add LIBERO-Safety benchmark as a submodule

Sibling to SafeLIBERO/, isolated for replicating arXiv:2606.23686's
eval environment without touching the existing benchmark setup.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds, `git submodule status` shows `LIBERO-Safety` initialized.

---

### Task 2: Download benchmark assets and isolate the LIBERO config

**Files:**
- Create: `libsafety_download_assets.sh`
- Create: `LIBERO-Safety/.libero_config.yaml` (generated at runtime by the libero package on first import, not hand-written — verified, not created by us)

**Interfaces:**
- Consumes: `LIBERO-Safety/` from Task 1.
- Produces: `LIBERO-Safety/libero/libero/assets/` populated; a documented `LIBSAFETY_CONFIG_PATH` env var pattern that later tasks source before running anything LIBERO-related, so it never collides with `SafeLIBERO`'s `~/.libero/config.yaml`.

- [ ] **Step 1: Write the asset-download script**

```bash
cat > /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_download_assets.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents"
LIBSAFETY_DIR="${REPO_ROOT}/LIBERO-Safety"

# Requires: pip install -U "huggingface_hub[cli]" in whichever env runs this
# (any Python env with huggingface_hub is fine; this step does not need
# torch/jax).
huggingface-cli download LIBERO-Safety/libero_safety_assets \
  --repo-type dataset \
  --local-dir "${LIBSAFETY_DIR}/_hf_assets_download"

# The HF repo ships assets.zip per the project README; unzip into the
# location the repo's libero package expects.
mkdir -p "${LIBSAFETY_DIR}/libero/libero/assets"
find "${LIBSAFETY_DIR}/_hf_assets_download" -iname "*.zip" -exec \
  unzip -o -q {} -d "${LIBSAFETY_DIR}/libero/libero/assets" \;

echo "Assets installed under ${LIBSAFETY_DIR}/libero/libero/assets"
ls "${LIBSAFETY_DIR}/libero/libero/assets" | head -20
EOF
chmod +x /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_download_assets.sh
```

Expected: script file created and executable.

- [ ] **Step 2: Run it (needs a Python env with `huggingface_hub` — the base conda env has it, or install with `pip install -U huggingface_hub` first)**

```bash
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
bash libsafety_download_assets.sh
```

Expected: exits 0, `LIBERO-Safety/libero/libero/assets/` is non-empty (`ls` shows asset subfolders, not an empty dir).

- [ ] **Step 3: Confirm LIBERO config isolation is documented for downstream tasks**

Every later task that imports `libero` must run with:

```bash
export LIBERO_CONFIG_PATH="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/LIBERO-Safety/.libero_config.yaml"
```

Verify this does not exist yet (so we know it'll be freshly generated on first `import libero`, pointing at `LIBERO-Safety/` paths, not `SafeLIBERO/`):

```bash
ls /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/LIBERO-Safety/.libero_config.yaml 2>&1
```

Expected: `No such file or directory` (confirms nothing pre-existing to collide with; Task 4's import smoke test will generate it correctly-scoped).

- [ ] **Step 4: Commit**

```bash
git add libsafety_download_assets.sh
git commit -m "$(cat <<'EOF'
feat: add LIBERO-Safety asset download script

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Build `libsafety_openpi` env (pi0 / pi0.5 policy server)

**Files:**
- Create: `libsafety_openpi_setup.sh`

**Interfaces:**
- Produces: conda env at `/ocean/projects/cis250185p/asingal/envs/libsafety_openpi` with `openpi` importable and `jax.devices()` reporting a GPU. Mirrors `openpi_setup.sh` / the existing `aegis` env, but fully separate.

- [ ] **Step 1: Write the setup script**

```bash
cat > /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_openpi_setup.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/libsafety_openpi"
OPENPI_REPO="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/vlsa-aegis/openpi"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -p "${ENV_PREFIX}" python=3.11 -y
conda activate "${ENV_PREFIX}"

pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu124
pip install "jax[cuda12]"==0.5.3

pip install -e "${OPENPI_REPO}"
pip install -e "${OPENPI_REPO}/packages/openpi-client"

pip install \
  flax==0.10.2 optax==0.2.4 orbax-checkpoint==0.11.13 chex==0.1.89 \
  equinox==0.12.2 augmax==0.4.1 \
  diffusers==0.33.1 transformers==4.53.2 safetensors==0.5.3 \
  tokenizers==0.21.1 sentencepiece==0.2.0 \
  opencv-python-headless==4.11.0.86 pillow==11.2.1 \
  numpy==1.26.4 scipy==1.15.3 einops==0.8.1 \
  wandb==0.19.11 rerun-sdk==0.23.1 \
  lerobot@git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 \
  draccus==0.10.0 tyro==0.9.22 omegaconf==2.3.0 \
  flask==3.1.1 websockets==15.0.1 \
  huggingface-hub==0.32.3 datasets==3.6.0 \
  av==14.4.0 imageio==2.37.0 imageio-ffmpeg==0.6.0 \
  dm-control==1.0.14 mujoco==2.3.7 \
  zarr==3.0.8 h5py==3.13.0 \
  rich==14.0.0 tqdm==4.67.1 \
  pydantic==2.11.5 jaxtyping==0.2.36 beartype==0.19.0

python -c "import jax; print('jax devices:', jax.devices())"
python -c "import openpi; print('openpi: OK')"
EOF
chmod +x /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_openpi_setup.sh
```

Expected: script created.

- [ ] **Step 2: Run on a GPU node**

```bash
salloc -J libsafety_setup -p GPU --gres=gpu:h100-80:1 --time=01:00:00
bash /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_openpi_setup.sh
```

Expected: final two `python -c` lines print `jax devices: [CudaDevice(id=0)...]` and `openpi: OK`. If they don't, fix the failing pip install and re-run (this is the "test" for this task — no pytest suite exists for env setup).

- [ ] **Step 3: Commit**

```bash
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
git add libsafety_openpi_setup.sh
git commit -m "$(cat <<'EOF'
feat: add libsafety_openpi env setup script for pi0/pi0.5 server

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Build `libsafety_client` env (LIBERO-Safety rollout loop for pi0/pi0.5)

**Files:**
- Create: `libsafety_client_setup.sh`

**Interfaces:**
- Consumes: `LIBERO-Safety/` (Task 1), assets (Task 2).
- Produces: conda env at `/ocean/projects/cis250185p/asingal/envs/libsafety_client` with `robosuite`, `libero` (from `LIBERO-Safety/`), and `openpi_client` importable. Mirrors the existing `libero_env`.

- [ ] **Step 1: Write the setup script**

```bash
cat > /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_client_setup.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/libsafety_client"
LIBSAFETY_DIR="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/LIBERO-Safety"
OPENPI_CLIENT_DIR="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/vlsa-aegis/openpi/packages/openpi-client"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -p "${ENV_PREFIX}" python=3.8.13 -y
conda activate "${ENV_PREFIX}"

pip install --upgrade pip

pip install \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

conda install -c conda-forge "av>=9.0.0" -y --freeze-installed

# Editable install of the LIBERO-Safety repo itself (per its own README:
# pip install -e ., then requirements.txt, extra_requirements.txt, robosuite)
cd "${LIBSAFETY_DIR}"
pip install -e .
pip install -r requirements.txt
pip install -r extra_requirements.txt
pip install -e third_party/robosuite-1.4

pip install -e "${OPENPI_CLIENT_DIR}" --no-deps
pip install "cachetools>=4.2.2" "httpx>=0.23.0" "websockets==13.1"

export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="${LIBSAFETY_DIR}/.libero_config.yaml"
python -c "
import robosuite; print('robosuite: OK')
from libero.libero.envs import OffScreenRenderEnv; print('libero envs: OK')
import openpi_client; print('openpi_client: OK')
"
EOF
chmod +x /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_client_setup.sh
```

Expected: script created.

- [ ] **Step 2: Run on a GPU node (needs `MUJOCO_GL=egl` for the render smoke test)**

```bash
export MUJOCO_GL=egl
bash /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_client_setup.sh
```

Expected: prints `robosuite: OK`, `libero envs: OK`, `openpi_client: OK`. If `pip install -e .`/`requirements.txt` fails on a specific pin (likely, given LIBERO-Safety's own requirements weren't fully visible ahead of time), resolve the conflict the same way `libero_env_setup.sh` did (pin/relax the offending package) and re-run until all three imports succeed.

- [ ] **Step 3: Verify `LIBERO-Safety/.libero_config.yaml` was generated and points into `LIBERO-Safety/`, not `SafeLIBERO/`**

```bash
cat /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/LIBERO-Safety/.libero_config.yaml
```

Expected: `assets`, `bddl_files`, `benchmark_root`, `datasets` all resolve under `.../LIBERO-Safety/libero/...`, confirming isolation from `SafeLIBERO/`.

- [ ] **Step 4: Commit**

```bash
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
git add libsafety_client_setup.sh
git commit -m "$(cat <<'EOF'
feat: add libsafety_client env setup script for LIBERO-Safety rollouts

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Build `libsafety_openvla` env (OpenVLA + OpenVLA-OFT, single process)

**Files:**
- Create: `libsafety_openvla_setup.sh`

**Interfaces:**
- Consumes: `LIBERO-Safety/` (Task 1), assets (Task 2), local `openvla-oft/` repo (already in this project, read-only).
- Produces: conda env at `/ocean/projects/cis250185p/asingal/envs/libsafety_openvla` with `torch`, `transformers`, `openvla-oft`, `robosuite`, and `libero` (from `LIBERO-Safety/`) importable. Mirrors `openvla_libero_merged`.

- [ ] **Step 1: Write the setup script**

```bash
cat > /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_openvla_setup.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="/ocean/projects/cis250185p/asingal/envs/libsafety_openvla"
LIBSAFETY_DIR="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/LIBERO-Safety"
OPENVLA_OFT_REPO="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/openvla-oft"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -p "${ENV_PREFIX}" python=3.10 -y
conda activate "${ENV_PREFIX}"

pip install --upgrade pip setuptools wheel

pip install \
  torch==2.2.0 \
  torchvision==0.17.0 \
  torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu118

pip install \
  accelerate>=0.25.0 \
  draccus==0.8.0 \
  einops==0.4.1 \
  huggingface_hub \
  json-numpy \
  jsonlines \
  matplotlib \
  peft==0.11.1 \
  protobuf \
  rich \
  sentencepiece==0.1.99 \
  timm==0.9.10 \
  tokenizers==0.19.1 \
  wandb \
  tensorflow==2.15.0 \
  tensorflow_datasets==4.9.3 \
  tensorflow_graphics==2021.12.3 \
  diffusers==0.30.3 \
  imageio \
  uvicorn \
  fastapi \
  packaging \
  ninja

cd "${LIBSAFETY_DIR}"
pip install -e .
pip install -r requirements.txt
pip install -r extra_requirements.txt
pip install -e third_party/robosuite-1.4

cd "${OPENVLA_OFT_REPO}"
pip install -e .

pip install tensorflow-addons==0.23.0
pip install "typeguard>=4,<5"
pip install tyro==0.9.2

export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="${LIBSAFETY_DIR}/.libero_config.yaml"
python -c "
import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
import robosuite; print('robosuite: OK')
import libero; print('libero: OK')
from experiments.robot.robot_utils import get_model; print('openvla-oft robot_utils: OK')
"
EOF
chmod +x /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_openvla_setup.sh
```

Expected: script created.

- [ ] **Step 2: Run on a GPU node**

```bash
export MUJOCO_GL=egl
bash /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_openvla_setup.sh
```

Expected: prints `Torch: 2.2.0+cu118 CUDA: True`, `robosuite: OK`, `libero: OK`, `openvla-oft robot_utils: OK`. Resolve any version conflicts the same way `openvla_safelibero_setup.sh`'s trailing "Error fixes" section does, and re-run.

- [ ] **Step 3: Confirm base (non-OFT) `openvla` is importable too — it ships inside `transformers`' `trust_remote_code` model hub path, not a separate pip package**

```bash
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openvla
python -c "
from transformers import AutoModelForVision2Seq, AutoProcessor
print('transformers AutoModelForVision2Seq/AutoProcessor: OK (used to load openvla/openvla-7b-finetuned-libero-* via trust_remote_code)')
"
```

Expected: prints the OK line with no import error.

- [ ] **Step 4: Commit**

```bash
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
git add libsafety_openvla_setup.sh
git commit -m "$(cat <<'EOF'
feat: add libsafety_openvla env setup script for OpenVLA/OpenVLA-OFT

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Download the four off-the-shelf checkpoints

**Files:**
- Create: `libsafety_download_checkpoints.sh`

**Interfaces:**
- Consumes: `libsafety_openpi` env (Task 3) for `gsutil`/HF auth if needed.
- Produces: `LIBERO-Safety/checkpoints/pi0_libero/`, `LIBERO-Safety/checkpoints/pi05_libero/` (openpi format), plus documents the two HF model IDs used directly by the eval script in Task 7 (no local download needed for those — `AutoModelForVision2Seq.from_pretrained` pulls them at eval time and HF caches them).

- [ ] **Step 1: Write the checkpoint download script (openpi checkpoints only — HF ones are pulled lazily by `from_pretrained`)**

```bash
cat > /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_download_checkpoints.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

LIBSAFETY_DIR="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/LIBERO-Safety"
mkdir -p "${LIBSAFETY_DIR}/checkpoints"

pip install --quiet gsutil || true

gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_libero "${LIBSAFETY_DIR}/checkpoints/"
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero "${LIBSAFETY_DIR}/checkpoints/"

echo "openpi checkpoints:"
ls "${LIBSAFETY_DIR}/checkpoints"

echo ""
echo "HF checkpoints (openvla, openvla-oft) are NOT pre-downloaded here —"
echo "run_libsafety_eval_openvla.py pulls them lazily via from_pretrained() into"
echo "the shared HF cache the first time each --pretrained_checkpoint is used:"
echo "  openvla/openvla-7b-finetuned-libero-spatial"
echo "  openvla/openvla-7b-finetuned-libero-object"
echo "  openvla/openvla-7b-finetuned-libero-goal"
echo "  openvla/openvla-7b-finetuned-libero-10"
echo "  moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
EOF
chmod +x /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_download_checkpoints.sh
```

Expected: script created.

- [ ] **Step 2: Run it**

```bash
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openpi
bash /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_download_checkpoints.sh
```

Expected: `LIBERO-Safety/checkpoints/pi0_libero/` and `.../pi05_libero/` both non-empty (`ls` output lists param/checkpoint files, not "No such file or directory").

- [ ] **Step 3: Commit**

```bash
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
git add libsafety_download_checkpoints.sh
git commit -m "$(cat <<'EOF'
feat: add checkpoint download script for pi0/pi0.5 openpi weights

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

(Note: `LIBERO-Safety/checkpoints/` must already be excluded from git tracking — verify in Step 4.)

- [ ] **Step 4: Verify checkpoints are gitignored, not staged**

```bash
git status --short LIBERO-Safety/checkpoints 2>&1
```

Expected: empty output (this repo's top-level `.gitignore` already excludes `checkpoints/`, and it applies inside the submodule's own working tree the same way `SafeLIBERO/`'s checkpoints are excluded — if it does show as untracked, add `checkpoints/` to `LIBERO-Safety/.git/info/exclude` before proceeding, never to the submodule's tracked `.gitignore`).

---

### Task 7: OpenVLA / OpenVLA-OFT eval driver + smoke test

**Files:**
- Create: `run_libsafety_eval_openvla.py`
- Test: manual smoke run (see Step 3) — this is an eval script, not a unit-testable library; "passing" means one episode completes and produces a video + metrics file.

**Interfaces:**
- Consumes: `libsafety_openvla` env (Task 5), checkpoints (Task 6), the real task-suite name(s) discovered in Task 1 Step 3.
- Produces: `LIBERO-Safety/rollouts/<run_id>/*.mp4` and `LIBERO-Safety/results/<run_id>.json` with `{task, episode, success, num_steps}` — same convention `run_libero_eval.py` uses in this repo (JSON summary + per-episode mp4).

- [ ] **Step 1: Write the eval driver, adapted from `run_libero_eval.py`**

```python
cat > /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/run_libsafety_eval_openvla.py << 'PYEOF'
"""
run_libsafety_eval_openvla.py

Eval-only driver for OpenVLA / OpenVLA-OFT on the LIBERO-Safety benchmark,
using standard off-the-shelf LIBERO-pretrained checkpoints (not the paper's
own fine-tuned weights). Mirrors run_libero_eval.py's structure but points
at the LIBERO-Safety task-suite registry instead of SafeLIBERO's.

Run inside the libsafety_openvla conda env:
  conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openvla
  export MUJOCO_GL=egl
  export LIBERO_CONFIG_PATH=.../LIBERO-Safety/.libero_config.yaml
  python run_libsafety_eval_openvla.py --pretrained_checkpoint <hf_id> \
      --task_suite_name <suite_name_from_Task1_Step3> --num_trials_per_task 1
"""

import json
import logging
import pathlib
import sys
from dataclasses import dataclass, field
from typing import List

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

sys.path.insert(0, str(pathlib.Path(__file__).parent / "openvla-oft"))
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    save_rollout_video,
)
from experiments.robot.robot_utils import get_action, get_image_resize_size, get_model, set_seed_everywhere

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class LibSafetyEvalConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: str = "openvla/openvla-7b-finetuned-libero-spatial"
    task_suite_name: str = ""  # fill in with a name confirmed from Task 1 Step 3
    task_index: int = 0
    num_trials_per_task: int = 1
    num_steps_wait: int = 10
    max_steps: int = 300
    center_crop: bool = True
    seed: int = 7
    video_out_path: str = "LIBERO-Safety/rollouts"
    results_out_path: str = "LIBERO-Safety/results"


@draccus.wrap()
def eval_libsafety(cfg: LibSafetyEvalConfig) -> None:
    assert cfg.task_suite_name, "task_suite_name must be set to a real LIBERO-Safety suite name"
    set_seed_everywhere(cfg.seed)

    model = get_model(cfg)
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.task_index)
    initial_states = task_suite.get_task_init_states(cfg.task_index)

    env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

    video_dir = pathlib.Path(cfg.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    results_dir = pathlib.Path(cfg.results_out_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    episode_results = []
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])
        replay_images = []
        t, done, success = 0, False, False

        while t < cfg.max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            img = get_libero_image(obs, resize_size)
            replay_images.append(img)
            action = get_action(cfg, model, obs, task_description)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

        save_rollout_video(
            replay_images,
            episode_idx,
            success=success,
            task_description=task_description,
            log_file=None,
        )
        episode_results.append({"task": task_description, "episode": episode_idx, "success": success, "num_steps": t})
        logger.info(f"Episode {episode_idx}: success={success}, steps={t}")

    with open(results_dir / f"{cfg.task_suite_name}_task{cfg.task_index}.json", "w") as f:
        json.dump(episode_results, f, indent=2)


if __name__ == "__main__":
    eval_libsafety()
PYEOF
```

Expected: file created. Note the deliberately empty `task_suite_name: str = ""` default with an `assert` — the real suite name string is discovered in Task 1 Step 3 by inspecting the cloned repo, not guessed here; the smoke test in Step 3 passes it explicitly on the command line.

- [ ] **Step 2: Confirm the real task-suite name before running**

```bash
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_client
export LIBERO_CONFIG_PATH=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/LIBERO-Safety/.libero_config.yaml
python -c "
from libero.libero import benchmark
d = benchmark.get_benchmark_dict()
print(list(d.keys()))
"
```

Expected: prints a Python list of real suite-name strings from `LIBERO-Safety`. Use the first one verbatim as `<suite_name>` in Step 3.

- [ ] **Step 3: Smoke test with one episode**

```bash
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openvla
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/LIBERO-Safety/.libero_config.yaml
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
python run_libsafety_eval_openvla.py \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name <suite_name_from_Step_2> \
    --task_index 0 \
    --num_trials_per_task 1
```

Expected: exits 0, logs `Episode 0: success=<True|False>, steps=<n>`, and produces one `.mp4` under `LIBERO-Safety/rollouts/` plus `LIBERO-Safety/results/<suite>_task0.json` containing the one episode record.

- [ ] **Step 4: Repeat Step 3 with the OFT checkpoint to confirm both share this driver**

```bash
python run_libsafety_eval_openvla.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name <suite_name_from_Step_2> \
    --task_index 0 \
    --num_trials_per_task 1
```

Expected: same success criteria as Step 3.

- [ ] **Step 5: Commit**

```bash
git add run_libsafety_eval_openvla.py
git commit -m "$(cat <<'EOF'
feat: add OpenVLA/OpenVLA-OFT eval driver for LIBERO-Safety

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: pi0 / pi0.5 server + client eval driver + smoke test

**Files:**
- Create: `libsafety_serve_openpi.sh`
- Create: `run_libsafety_eval_openpi.py`

**Interfaces:**
- Consumes: `libsafety_openpi` env (Task 3), `libsafety_client` env (Task 4), checkpoints (Task 6), suite name (Task 7 Step 2).
- Produces: same `LIBERO-Safety/rollouts/` + `LIBERO-Safety/results/` output convention as Task 7.

- [ ] **Step 1: Write the server launch script**

```bash
cat > /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_serve_openpi.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
# Usage: bash libsafety_serve_openpi.sh <pi0_libero|pi05_libero>
POLICY_CONFIG="${1:?pass pi0_libero or pi05_libero}"
LIBSAFETY_DIR="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/LIBERO-Safety"

conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONPATH="${PYTHONPATH:-}:${LIBSAFETY_DIR}/../vlsa-aegis/openpi/src"

python /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/vlsa-aegis/openpi/scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config="${POLICY_CONFIG}" \
    --policy.dir="${LIBSAFETY_DIR}/checkpoints/${POLICY_CONFIG}"
EOF
chmod +x /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_serve_openpi.sh
```

Expected: script created.

- [ ] **Step 2: Write the client eval driver, adapted from `vlsa-aegis/main/pi05_evaluation.py`**

```python
cat > /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/run_libsafety_eval_openpi.py << 'PYEOF'
"""
run_libsafety_eval_openpi.py

Eval-only client driver for pi0 / pi0.5 on the LIBERO-Safety benchmark.
Connects to a running libsafety_serve_openpi.sh policy server over
websocket. Mirrors vlsa-aegis/main/pi05_evaluation.py's structure but
points at the LIBERO-Safety task-suite registry instead of SafeLIBERO's.

Run inside the libsafety_client conda env, after the server is up:
  conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_client
  export MUJOCO_GL=egl
  export LIBERO_CONFIG_PATH=.../LIBERO-Safety/.libero_config.yaml
  python run_libsafety_eval_openpi.py --task_suite_name <suite_name> \
      --task_index 0 --num_trials_per_task 1
"""

import dataclasses
import json
import logging
import pathlib

import numpy as np
import tqdm
import tyro
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = ""  # fill in with a name confirmed via benchmark.get_benchmark_dict()
    task_index: int = 0
    num_trials_per_task: int = 1
    num_steps_wait: int = 10
    max_steps: int = 300
    video_out_path: str = "LIBERO-Safety/rollouts"
    results_out_path: str = "LIBERO-Safety/results"
    seed: int = 7


def _get_env(task, resolution, seed):
    from libero.libero import get_libero_path
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def eval_libsafety(args: Args) -> None:
    assert args.task_suite_name, "task_suite_name must be set to a real LIBERO-Safety suite name"
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_index)
    initial_states = task_suite.get_task_init_states(args.task_index)
    env, task_description = _get_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    video_dir = pathlib.Path(args.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    results_dir = pathlib.Path(args.results_out_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    episode_results = []
    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])
        t, done, success = 0, False, False
        replay_images = []

        while t < args.max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["agentview_image"], args.resize_size, args.resize_size)
            )
            replay_images.append(img)
            result = client.infer({"observation/image": img, "prompt": task_description})
            action = np.array(result["actions"][0])
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

        episode_results.append({"task": task_description, "episode": episode_idx, "success": success, "num_steps": t})
        logger.info(f"Episode {episode_idx}: success={success}, steps={t}")

    with open(results_dir / f"{args.task_suite_name}_task{args.task_index}.json", "w") as f:
        json.dump(episode_results, f, indent=2)


if __name__ == "__main__":
    tyro.cli(eval_libsafety, args=(Args(),))
PYEOF
```

Expected: file created. `task_suite_name` again deliberately empty with an assert, filled at run time with the real value from Task 7 Step 2.

- [ ] **Step 3: Smoke test — start the pi0.5 server, then run the client in a second shell**

```bash
# Shell A (server):
bash /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_serve_openpi.sh pi05_libero
# wait for: "INFO:websockets.server:server listening on 0.0.0.0:8000"
```

```bash
# Shell B (client), once the server log shows "listening":
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_client
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/LIBERO-Safety/.libero_config.yaml
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
python run_libsafety_eval_openpi.py \
    --task_suite_name <suite_name_from_Task7_Step2> \
    --task_index 0 \
    --num_trials_per_task 1
```

Expected: client exits 0, logs `Episode 0: success=<True|False>, steps=<n>`, writes `LIBERO-Safety/results/<suite>_task0.json`. Stop the server (`Ctrl+C` in Shell A).

- [ ] **Step 4: Repeat Step 3 with `pi0_libero` instead of `pi05_libero`**

```bash
bash /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/libsafety_serve_openpi.sh pi0_libero
# then, in Shell B, same client command as Step 3
```

Expected: same success criteria as Step 3.

- [ ] **Step 5: Commit**

```bash
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
git add libsafety_serve_openpi.sh run_libsafety_eval_openpi.py
git commit -m "$(cat <<'EOF'
feat: add pi0/pi0.5 server + eval driver for LIBERO-Safety

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Push branch and final cross-policy verification

**Files:** none new — verification only.

**Interfaces:**
- Consumes: everything from Tasks 1-8.

- [ ] **Step 1: Confirm all four policies have a successful smoke-test JSON result on disk**

```bash
ls /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/LIBERO-Safety/results/
```

Expected: at least one `*_task0.json` per policy run performed in Tasks 7-8 (four total files if suite/task combos were kept the same across runs, or more if varied — the point is each of the 4 policies produced at least one).

- [ ] **Step 2: Confirm existing envs/branches are untouched**

```bash
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
git status SafeLIBERO openvla-oft vlsa-aegis
conda env list | grep -E "aegis|libero_env|openvla_libero_merged"
```

Expected: `git status` on those three paths shows no changes beyond whatever was already dirty before this plan started (compare against the state noted at Task 1 Step 1); the three original conda envs are still listed unchanged.

- [ ] **Step 3: Push the branch**

```bash
git push origin libero-safety-benchmark
```

Expected: new branch appears on `origin` (`AkankshaSingal8/situational_safety_for_embodied_agents`), no changes pushed to `full_working_pipeline` or `main`.

---

## Self-Review Notes

- Spec coverage: branch/submodule isolation (Task 1), asset download + config isolation (Task 2), three-env split mirroring the existing SafeLIBERO pattern (Tasks 3-5), off-the-shelf-checkpoint-only strategy (Task 6, explicit HF IDs, explicit note that paper's own checkpoints are not used), eval drivers mirroring `run_libero_eval.py` / `pi05_evaluation.py` (Tasks 7-8), validation via smoke episodes for all four policies (Tasks 7-9) — all spec sections have a corresponding task.
- The LIBERO-Safety task-suite name strings are the one piece of information not yet knowable from outside the cloned repo; every task that needs one explicitly discovers it via `benchmark.get_benchmark_dict()` (Task 7 Step 2) rather than guessing a fabricated name, and downstream commands reference that discovered value by name.
