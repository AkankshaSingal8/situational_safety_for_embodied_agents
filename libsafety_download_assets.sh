#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents"
LIBSAFETY_DIR="${REPO_ROOT}/LIBERO-Safety"

# Requires: pip install -U "huggingface_hub[cli]" in whichever env runs this
# (any Python env with huggingface_hub is fine; this step does not need
# torch/jax).
#
# NOTE: `huggingface-cli` is deprecated in current huggingface_hub releases
# in favor of the `hf` command (same functionality). We use `hf download`
# here; if your huggingface_hub version predates the `hf` CLI, substitute
# `huggingface-cli download` with the same arguments.
hf download LIBERO-Safety/libero_safety_assets \
  --repo-type dataset \
  --local-dir "${LIBSAFETY_DIR}/_hf_assets_download"

# The HF repo ships assets.zip per the project README. The zip's internal
# structure has a top-level "assets/" directory, so per the README we unzip
# into LIBERO-Safety/libero/libero/ (NOT directly into .../assets/) so that
# the result lands at libero/libero/assets/. (Verified via `unzip -l` before
# writing this script: the archive's first entries are "assets/",
# "assets/articulated_objects/", etc.)
mkdir -p "${LIBSAFETY_DIR}/libero/libero"
find "${LIBSAFETY_DIR}/_hf_assets_download" -iname "*.zip" -exec \
  unzip -o -q {} -d "${LIBSAFETY_DIR}/libero/libero" \;

echo "Assets installed under ${LIBSAFETY_DIR}/libero/libero/assets"
ls "${LIBSAFETY_DIR}/libero/libero/assets" | head -20
