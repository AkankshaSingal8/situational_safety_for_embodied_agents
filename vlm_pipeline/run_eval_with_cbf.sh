#!/usr/bin/env bash
# run_eval_with_cbf.sh
#
# Evaluate OpenVLA policy on SafeLIBERO using pre-computed CBF ellipsoids.
# VLM inference is NOT run here — CBF parameters are loaded from cbf_outputs/.
# Run qwen_vlm_worker.py and cbf_construction.py first to generate CBF outputs.
#
# Requires: openvla_libero_merged conda env + GPU
#
# Usage:
#   conda activate openvla_libero_merged
#   export MUJOCO_GL=egl
#   bash run_eval_with_cbf.sh

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

# Output
RUN_NOTE="${RUN_NOTE:-}"                         # Optional suffix appended to run_id

# ── Run ───────────────────────────────────────────────────────────────────────

echo "=== Eval with Pre-computed CBF ==="
echo "  checkpoint:       $CHECKPOINT"
echo "  task_suite:       $TASK_SUITE  (safety_level=$SAFETY_LEVEL)"
echo "  num_trials:       $NUM_TRIALS"
echo "  use_safety_filter: $USE_SAFETY_FILTER"
echo ""

EXTRA_ARGS=()
[ -n "$RUN_NOTE" ] && EXTRA_ARGS+=(--run_id_note "$RUN_NOTE")

python run_libero_eval_with_cbf.py \
    --pretrained_checkpoint "$CHECKPOINT" \
    --model_family         "$MODEL_FAMILY" \
    --task_suite_name      "$TASK_SUITE" \
    --safety_level         "$SAFETY_LEVEL" \
    --num_trials_per_task  "$NUM_TRIALS" \
    --use_safety_filter    "$USE_SAFETY_FILTER" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Done ==="
