#!/usr/bin/env bash
# run_qwen_vlm.sh
#
# Run qwen_vlm_worker.py to extract semantic safety constraints from episode observations.
# Supports single-episode and batch modes, with optional dry_run (no GPU needed).
#
# Usage (single episode):
#   conda activate qwen
#   bash run_qwen_vlm.sh
#
# Usage (batch — set MODE=batch and INPUT_DIR below):
#   MODE=batch bash run_qwen_vlm.sh
#
# Usage (dry run, no GPU):
#   DRY_RUN=true bash run_qwen_vlm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configuration ─────────────────────────────────────────────────────────────

METHOD="${METHOD:-m1}"                    # m1 (Seg+VLM), m2 (VLM-only), m3 (3D+VLM)
MODEL="${MODEL:-qwen2.5-vl-7b}"           # Qwen model key
NUM_VOTES="${NUM_VOTES:-5}"               # Majority-voting rounds per query
DRY_RUN="${DRY_RUN:-false}"              # true = placeholder responses, no GPU needed

# Single-episode mode
MODE="${MODE:-single}"                    # single | batch
INPUT_FOLDER="${INPUT_FOLDER:-../vlm_inputs/safelibero_spatial/level_I/task_0/episode_00}"
OUTPUT_JSON="${OUTPUT_JSON:-../results/m1_task0_ep00.json}"

# Batch mode (used when MODE=batch)
INPUT_DIR="${INPUT_DIR:-../vlm_inputs/safelibero_spatial}"
BATCH_OUTPUT_JSON="${BATCH_OUTPUT_JSON:-../results/m1_spatial_all.json}"

# ── Build command ─────────────────────────────────────────────────────────────

CMD=(python qwen_vlm_worker.py
    --method "$METHOD"
    --model "$MODEL"
    --num_votes "$NUM_VOTES"
)

if [ "$MODE" = "batch" ]; then
    CMD+=(--input_dir "$INPUT_DIR" --output_json "$BATCH_OUTPUT_JSON")
else
    CMD+=(--input_folder "$INPUT_FOLDER" --output_json "$OUTPUT_JSON")
fi

if [ "$DRY_RUN" = "true" ]; then
    CMD+=(--dry_run)
fi

# ── Run ───────────────────────────────────────────────────────────────────────

echo "=== Qwen VLM Worker ==="
echo "  method:    $METHOD"
echo "  model:     $MODEL"
echo "  num_votes: $NUM_VOTES"
echo "  mode:      $MODE"
echo "  dry_run:   $DRY_RUN"
echo ""

"${CMD[@]}"

echo ""
echo "=== Done ==="
