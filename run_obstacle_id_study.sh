#!/usr/bin/env bash
# run_obstacle_id_study.sh
#
# Two-phase obstacle-identification prompt study.
#
# Phase 1: Run all 5 prompts with qwen3-vl-8b to find the best prompt.
# Phase 2: Run the best prompt with qwen2.5-vl-7b and qwen2.5-vl-3b.
#
# Usage (interactive node or SLURM job):
#   bash run_obstacle_id_study.sh
#
# Environment variables:
#   CONDA_ENV     conda env to activate (default: qwen)
#   PROJECT_ROOT  project root (default: directory containing this script)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
CONDA_ENV="${CONDA_ENV:-qwen}"

SUITE="safelibero_spatial"
LEVEL="I"
TASK="0"
PROMPTS_DIR="prompts/obstacle_id"
RESULTS_DIR="vlm_prompt_runner/results"

# Activate conda if not already active
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    echo "[study] Activating conda env: $CONDA_ENV"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
cd "$PROJECT_ROOT"
mkdir -p "$RESULTS_DIR"

# ── Phase 1: Find best prompt with qwen3-vl-8b ──────────────────────────────
echo ""
echo "=== Phase 1: All prompts × qwen3-vl-8b ==="
python -m vlm_prompt_runner.run_prompt_experiment \
    --prompts-dir "$PROMPTS_DIR" \
    --model qwen3-vl-8b \
    --suite "$SUITE" --level "$LEVEL" --task "$TASK" \
    --results-out "$RESULTS_DIR/phase1_qwen3_vl_8b.json"

BEST_PROMPT=$(python -c "
import json, sys
d = json.load(open('$RESULTS_DIR/phase1_qwen3_vl_8b.json'))
print(d['best_prompt'])
")
echo ""
echo "[study] Best prompt from Phase 1: $BEST_PROMPT"

# ── Phase 2: Best prompt × remaining models ──────────────────────────────────
echo ""
echo "=== Phase 2: $BEST_PROMPT × qwen2.5-vl-7b and qwen2.5-vl-3b ==="
for MODEL in qwen2.5-vl-7b qwen2.5-vl-3b; do
    MODEL_SLUG="${MODEL//./_}"
    echo ""
    echo "--- Model: $MODEL ---"
    python -m vlm_prompt_runner.run_prompt_experiment \
        --prompts-dir "$PROMPTS_DIR" \
        --prompts "$BEST_PROMPT" \
        --model "$MODEL" \
        --suite "$SUITE" --level "$LEVEL" --task "$TASK" \
        --results-out "$RESULTS_DIR/phase2_${MODEL_SLUG}.json"
done

# ── Final comparison table ───────────────────────────────────────────────────
echo ""
echo "=== Final Cross-Model Comparison ==="
python - <<'PYEOF'
import json
from pathlib import Path

results_dir = Path("vlm_prompt_runner/results")
p1 = json.loads((results_dir / "phase1_qwen3_vl_8b.json").read_text())
best = p1["best_prompt"]

print(f"Best prompt (from Phase 1): {best}\n")
header = f"{'Model':<25} {'Prompt':<35} {'Correct':>8} {'Total':>6} {'Accuracy':>10}"
print(header)
print("-" * len(header))

# Phase 1 — best prompt row only
stats = p1["prompts"][best]
print(f"{'qwen3-vl-8b':<25} {best:<35} {stats['correct']:>8} {stats['total']:>6} {stats['accuracy']:>9.1%}")

# Phase 2
for fname in sorted(results_dir.glob("phase2_*.json")):
    d = json.loads(fname.read_text())
    for stem, s in d["prompts"].items():
        print(f"{d['model']:<25} {stem:<35} {s['correct']:>8} {s['total']:>6} {s['accuracy']:>9.1%}")

# All Phase 1 prompts
print()
print("Phase 1 — all prompts (qwen3-vl-8b):")
for stem, stats in sorted(p1["prompts"].items(), key=lambda x: -x[1]["accuracy"]):
    marker = "  ← best" if stem == best else ""
    print(f"  {stem:<33} {stats['correct']:>3}/{stats['total']:>3}  ({stats['accuracy']:.1%}){marker}")
PYEOF
