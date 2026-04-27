#!/usr/bin/env bash
#SBATCH --job-name=obstacle-id-phase2
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/phase2_%j.out
#SBATCH --error=slurm/logs/phase2_%j.err

# Phase 2: Run the best prompt (from Phase 1) with qwen2.5-vl-7b and qwen2.5-vl-3b.
# Saves per-model results and prints a cross-model comparison table.
#
# Must run AFTER Phase 1 completes (needs phase1_qwen3_vl_8b.json).
#
# Submit after Phase 1:
#   PHASE1_JOB=<phase1_job_id>
#   sbatch --dependency=afterok:$PHASE1_JOB slurm/phase2_model_compare.sh
#
# Or manually after Phase 1 finishes:
#   sbatch slurm/phase2_model_compare.sh

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents"
CONDA_ENV="qwen"

echo "=== Phase 2: Best prompt × qwen2.5-vl-7b and qwen2.5-vl-3b ==="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Started     : $(date)"

# ── Environment ─────────────────────────────────────────────────────────────
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export MUJOCO_GL=egl
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"

cd "$PROJECT_ROOT"
mkdir -p vlm_prompt_runner/results slurm/logs

# ── Read best prompt from Phase 1 results ────────────────────────────────────
PHASE1_RESULTS="vlm_prompt_runner/results/phase1_qwen3_vl_8b.json"
if [[ ! -f "$PHASE1_RESULTS" ]]; then
    echo "ERROR: Phase 1 results not found at $PHASE1_RESULTS"
    echo "Run phase1_prompt_study.sh first."
    exit 1
fi

BEST_PROMPT=$(python -c "
import json
d = json.load(open('$PHASE1_RESULTS'))
print(d['best_prompt'])
")
echo "Best prompt from Phase 1: $BEST_PROMPT"
echo ""

# ── Run best prompt with each remaining model ─────────────────────────────────
for MODEL in qwen2.5-vl-7b qwen2.5-vl-3b; do
    MODEL_SLUG="${MODEL//./_}"
    echo "--- Model: $MODEL ---"

    python -m vlm_prompt_runner.run_prompt_experiment \
        --prompts-dir prompts/obstacle_id \
        --prompts "$BEST_PROMPT" \
        --model "$MODEL" \
        --suite safelibero_spatial \
        --level I \
        --task 0 \
        --output-base "vlm_prompt_runner/outputs/${MODEL_SLUG}" \
        --results-out "vlm_prompt_runner/results/phase2_${MODEL_SLUG}.json" \
        --max-new-tokens 1024

    echo ""
done

# ── Final cross-model comparison table ───────────────────────────────────────
python - <<'PYEOF'
import json
from pathlib import Path

results_dir = Path("vlm_prompt_runner/results")
p1 = json.loads((results_dir / "phase1_qwen3_vl_8b.json").read_text())
best = p1["best_prompt"]

print("=" * 85)
print("FINAL CROSS-MODEL COMPARISON")
print("=" * 85)
print(f"Best prompt (from Phase 1 / qwen3-vl-8b): {best}\n")

header = f"{'Model':<25} {'Prompt':<35} {'Correct':>8} {'Total':>6} {'Accuracy':>10}"
print(header)
print("-" * len(header))

stats = p1["prompts"][best]
print(f"{'qwen3-vl-8b':<25} {best:<35} {stats['correct']:>8} {stats['total']:>6} {stats['accuracy']:>9.1%}")

for fname in sorted(results_dir.glob("phase2_*.json")):
    d = json.loads(fname.read_text())
    for stem, s in d["prompts"].items():
        print(f"{d['model']:<25} {stem:<35} {s['correct']:>8} {s['total']:>6} {s['accuracy']:>9.1%}")

print()
print("Phase 1 — all prompts (qwen3-vl-8b):")
for stem, stats in sorted(p1["prompts"].items(), key=lambda x: -x[1]["accuracy"]):
    marker = "  ← best" if stem == best else ""
    print(f"  {stem:<33} {stats['correct']:>3}/{stats['total']:>3}  ({stats['accuracy']:.1%}){marker}")
PYEOF

echo ""
echo "Phase 2 complete: $(date)"
