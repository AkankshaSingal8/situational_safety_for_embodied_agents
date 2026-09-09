#!/usr/bin/env bash
# run_eval_integrated.sh
#
# Full end-to-end evaluation: OpenVLA policy + live Qwen VLM constraint extraction + CBF filter.
# The Qwen VLM runs as a persistent HTTP server (qwen_vlm_server.py) in a separate terminal/job
# before launching this script, unless VLM_DRY_RUN=true (smoke-test mode, no GPU needed for VLM).
#
# Requires: openvla_libero_merged conda env + GPU (A100 recommended)
#
# Usage:
#   conda activate openvla_libero_merged
#   export MUJOCO_GL=egl
#   bash run_eval_integrated.sh
#
# Dry-run smoke test (no Qwen server needed):
#   VLM_DRY_RUN=true bash run_eval_integrated.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

# ── Configuration ─────────────────────────────────────────────────────────────

# Policy
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10}"
MODEL_FAMILY="${MODEL_FAMILY:-openvla}"

# Task
TASK_SUITE="${TASK_SUITE:-safelibero_spatial}"   # safelibero_spatial | safelibero_goal
SAFETY_LEVEL="${SAFETY_LEVEL:-I}"                # I | II
NUM_TRIALS="${NUM_TRIALS:-50}"                   # Rollouts per task

# Safety filter
USE_SAFETY_FILTER="${USE_SAFETY_FILTER:-True}"

# VLM
VLM_METHOD="${VLM_METHOD:-m1}"                          # m1 | m2 | m3
VLM_DRY_RUN="${VLM_DRY_RUN:-False}"                     # True = placeholder responses, no server needed
VLM_SERVER_URL="${VLM_SERVER_URL:-http://localhost:5001}"
VLM_MODEL="${VLM_MODEL:-qwen2.5-vl-7b}"
NUM_VLM_VOTES="${NUM_VLM_VOTES:-1}"
QWEN_CONDA_ENV="${QWEN_CONDA_ENV:-qwen}"
VLM_TMP_DIR="${VLM_TMP_DIR:-/tmp/vlm_obs}"

# Output
RESULTS_DIR="${RESULTS_DIR:-../openvla_cbf_benchmark}"
VIDEO_DIR="${VIDEO_DIR:-../openvla_cbf_video}"
RUN_NOTE="${RUN_NOTE:-}"                                 # Optional suffix appended to run_id

# ── Run ───────────────────────────────────────────────────────────────────────

echo "=== Integrated Eval: OpenVLA + VLM CBF ==="
echo "  checkpoint:    $CHECKPOINT"
echo "  task_suite:    $TASK_SUITE  (safety_level=$SAFETY_LEVEL)"
echo "  num_trials:    $NUM_TRIALS"
echo "  vlm_method:    $VLM_METHOD  (dry_run=$VLM_DRY_RUN)"
echo "  results_dir:   $RESULTS_DIR"
echo ""

EXTRA_ARGS=()
[ -n "$RUN_NOTE" ] && EXTRA_ARGS+=(--run_id_note "$RUN_NOTE")

python run_libero_eval_integrated.py \
    --pretrained_checkpoint "$CHECKPOINT" \
    --model_family         "$MODEL_FAMILY" \
    --task_suite_name      "$TASK_SUITE" \
    --safety_level         "$SAFETY_LEVEL" \
    --num_trials_per_task  "$NUM_TRIALS" \
    --use_safety_filter    "$USE_SAFETY_FILTER" \
    --vlm_method           "$VLM_METHOD" \
    --vlm_dry_run          "$VLM_DRY_RUN" \
    --vlm_server_url       "$VLM_SERVER_URL" \
    --vlm_model            "$VLM_MODEL" \
    --num_vlm_votes        "$NUM_VLM_VOTES" \
    --qwen_conda_env       "$QWEN_CONDA_ENV" \
    --vlm_tmp_dir          "$VLM_TMP_DIR" \
    --results_output_dir   "$RESULTS_DIR" \
    --video_output_dir     "$VIDEO_DIR" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Done. Results in $RESULTS_DIR/$TASK_SUITE/ ==="
