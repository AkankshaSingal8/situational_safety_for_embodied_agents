#!/usr/bin/env bash
# run_smoke_test.sh
#
# Smoke test for run_libero_eval_integrated.py: 1 episode, dry_run=True (no Qwen GPU needed).
# Run from the project root on Bridges2 after activating openvla_libero_merged.
#
# Usage:
#   conda activate openvla_libero_merged
#   bash run_smoke_test.sh
#
# Expected output:
#   [VLM] t=0 chunk=0: 0 ellipsoids (dry_run returns empty constraints)
#   success=... collide=... steps=...
#   Results saved to /tmp/integrated_smoke/safelibero_spatial/results_EVAL-*.json

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export MUJOCO_GL=egl

# Clean up stale data from previous runs to ensure idempotency
rm -rf /tmp/vlm_smoke /tmp/integrated_smoke

echo "=== Smoke test: 1 episode, dry_run=True ==="
python run_libero_eval_integrated.py \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 1 \
    --use_safety_filter True \
    --vlm_dry_run True \
    --vlm_server_url "http://localhost:5001" \
    --vlm_tmp_dir /tmp/vlm_smoke \
    --results_output_dir /tmp/integrated_smoke \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10

echo ""
echo "=== Smoke test complete. Checking results... ==="
RESULTS_FILE=$(ls -t /tmp/integrated_smoke/safelibero_spatial/results_EVAL-*.json 2>/dev/null | head -1)
if [ -z "$RESULTS_FILE" ]; then
    echo "ERROR: No results file found in /tmp/integrated_smoke/safelibero_spatial/"
    exit 1
fi
echo "Results file: $RESULTS_FILE"
python -c "
import json, sys
with open('$RESULTS_FILE') as f:
    d = json.load(f)
overall = d.get('overall', {})
print(f'  TSR: {overall.get(\"TSR\", \"N/A\")}')
print(f'  CAR: {overall.get(\"CAR\", \"N/A\")}')
print(f'  ETS_mean: {overall.get(\"ETS_mean\", \"N/A\")}')
print(f'  vlm_dry_run: {overall.get(\"vlm_dry_run\", \"N/A\")}')
required = ['TSR', 'CAR', 'ETS_mean', 'vlm_dry_run']
missing = [k for k in required if k not in overall]
if missing:
    print(f'ERROR: Missing keys: {missing}')
    sys.exit(1)
print('All required keys present. Smoke test PASSED.')
"
