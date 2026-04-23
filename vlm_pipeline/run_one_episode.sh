#!/usr/bin/env bash
# run_one_episode.sh
#
# Run the integrated SafeLIBERO evaluation for exactly 1 episode per task.
# Default: dry_run=True (no Qwen GPU needed — good for pipeline testing).
# Set DRY_RUN=false to use the live Qwen server (server must be running first).
#
# Usage (dry run, no GPU for VLM):
#   conda activate openvla_libero_merged
#   bash run_one_episode.sh
#
# Usage (live VLM, Qwen server must be started separately):
#   conda activate /ocean/projects/cis250185p/asingal/envs/qwen
#   CUDA_VISIBLE_DEVICES=1 python qwen_vlm_server.py --port 5001 --model qwen2.5-vl-3b &
#   sleep 30
#   conda activate openvla_libero_merged
#   DRY_RUN=false bash run_one_episode.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ── Config (override with env vars) ──────────────────────────────────────────
SUITE="${SUITE:-safelibero_spatial}"
LEVEL="${LEVEL:-I}"
DRY_RUN="${DRY_RUN:-true}"
VLM_SERVER_URL="${VLM_SERVER_URL:-http://localhost:5001}"
VLM_MODEL="${VLM_MODEL:-qwen2.5-vl-3b}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/integrated_one_episode}"
VLM_TMP_DIR="${VLM_TMP_DIR:-/tmp/vlm_one_episode}"

module load cuda/11.7.1 2>/dev/null || true

export MUJOCO_GL=egl

# Clean up stale data from previous runs
rm -rf "${VLM_TMP_DIR}" "${RESULTS_DIR}"

echo "========================================================"
echo " SafeLIBERO Integrated Evaluation — 1 episode per task"
echo "========================================================"
echo "  Suite:        ${SUITE}"
echo "  Safety level: ${LEVEL}"
echo "  dry_run:      ${DRY_RUN}"
echo "  VLM server:   ${VLM_SERVER_URL}"
echo "  Results dir:  ${RESULTS_DIR}"
echo "========================================================"
echo ""

python run_libero_eval_integrated.py \
    --task_suite_name "${SUITE}" \
    --safety_level "${LEVEL}" \
    --num_trials_per_task 1 \
    --use_safety_filter True \
    --vlm_dry_run "${DRY_RUN}" \
    --vlm_server_url "${VLM_SERVER_URL}" \
    --vlm_model "${VLM_MODEL}" \
    --num_vlm_votes 1 \
    --vlm_tmp_dir "${VLM_TMP_DIR}" \
    --results_output_dir "${RESULTS_DIR}" \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --run_id_note "one_episode"

echo ""
echo "=== Results ==="
RESULTS_FILE=$(ls -t "${RESULTS_DIR}/${SUITE}/results_EVAL-"*.json 2>/dev/null | head -1)
if [ -z "${RESULTS_FILE}" ]; then
    echo "ERROR: No results file found under ${RESULTS_DIR}/${SUITE}/"
    exit 1
fi
echo "File: ${RESULTS_FILE}"
RESULTS_FILE="${RESULTS_FILE}" python - <<'PYEOF'
import json, os, sys
with open(os.environ["RESULTS_FILE"]) as f:
    d = json.load(f)
o = d.get("overall", {})
print(f"  TSR:          {o.get('TSR', 'N/A')}")
print(f"  CAR:          {o.get('CAR', 'N/A')}")
print(f"  ETS_mean:     {o.get('ETS_mean', 'N/A')}")
print(f"  Episodes:     {o.get('total_episodes', 'N/A')}")
print(f"  vlm_dry_run:  {o.get('vlm_dry_run', 'N/A')}")
PYEOF
