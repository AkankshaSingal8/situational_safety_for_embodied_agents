#!/usr/bin/env bash
# run_eval_openvla.sh
#
# Baseline OpenVLA-OFT evaluation on SafeLIBERO — no CBF safety filter.
# Use this to establish baseline task success and violation rates before
# comparing against the integrated or CBF-filtered pipelines.
#
# Requires: openvla_libero_merged conda env + GPU
#
# Usage:
#   conda activate openvla_libero_merged
#   export MUJOCO_GL=egl
#   bash run_eval_openvla.sh
#
# Override any variable inline, e.g.:
#   SAFETY_LEVEL=II NUM_TRIALS=20 bash run_eval_openvla.sh

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

# Output
VIDEO_DIR="${VIDEO_DIR:-../openvla_video}"
RESULTS_DIR="${RESULTS_DIR:-../openvla_benchmark}"
RUN_NOTE="${RUN_NOTE:-}"                         # Optional suffix appended to run_id

# ── Run ───────────────────────────────────────────────────────────────────────

echo "=== Baseline OpenVLA-OFT Eval (no safety filter) ==="
echo "  checkpoint:  $CHECKPOINT"
echo "  task_suite:  $TASK_SUITE  (safety_level=$SAFETY_LEVEL)"
echo "  num_trials:  $NUM_TRIALS"
echo "  results_dir: $RESULTS_DIR"
echo ""

EXTRA_ARGS=()
[ -n "$RUN_NOTE" ] && EXTRA_ARGS+=(--run_id_note "$RUN_NOTE")

python run_safelibero_openvla_oft_eval.py \
    --pretrained_checkpoint "$CHECKPOINT" \
    --model_family         "$MODEL_FAMILY" \
    --task_suite_name      "$TASK_SUITE" \
    --safety_level         "$SAFETY_LEVEL" \
    --num_trials_per_task  "$NUM_TRIALS" \
    --video_output_dir     "$VIDEO_DIR" \
    --results_output_dir   "$RESULTS_DIR" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Done. Results in $RESULTS_DIR/$TASK_SUITE/ ==="
