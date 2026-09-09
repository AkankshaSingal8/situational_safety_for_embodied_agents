#!/usr/bin/env bash
set -euo pipefail

LIBSAFETY_DIR="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/LIBERO-Safety"
mkdir -p "${LIBSAFETY_DIR}/checkpoints"

pip install --quiet gsutil || true

gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero "${LIBSAFETY_DIR}/checkpoints/"
gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_libero "${LIBSAFETY_DIR}/checkpoints/"

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
