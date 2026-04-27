#!/usr/bin/env bash
#SBATCH --job-name=obstacle-id-phase1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/phase1_%j.out
#SBATCH --error=slurm/logs/phase1_%j.err

# Phase 1: Run all 5 obstacle-id prompts with qwen3-vl-8b.
# Finds the best-performing prompt and saves results to
# vlm_prompt_runner/results/phase1_qwen3_vl_8b.json
#
# Submit:
#   mkdir -p slurm/logs
#   sbatch slurm/phase1_prompt_study.sh
#
# Then submit Phase 2 with dependency:
#   sbatch --dependency=afterok:$SLURM_JOB_ID slurm/phase2_model_compare.sh

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents"
CONDA_ENV="qwen"

echo "=== Phase 1: All prompts × qwen3-vl-8b ==="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Started     : $(date)"
echo "Project root: $PROJECT_ROOT"

# ── Environment ─────────────────────────────────────────────────────────────
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export MUJOCO_GL=egl
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"

cd "$PROJECT_ROOT"
mkdir -p vlm_prompt_runner/results slurm/logs

# ── Run experiment ───────────────────────────────────────────────────────────
python -m vlm_prompt_runner.run_prompt_experiment \
    --prompts-dir prompts/obstacle_id \
    --model qwen3-vl-8b \
    --suite safelibero_spatial \
    --level I \
    --task 0 \
    --output-base vlm_prompt_runner/outputs/qwen3_vl_8b \
    --results-out vlm_prompt_runner/results/phase1_qwen3_vl_8b.json \
    --max-new-tokens 1024

# ── Print best prompt so it appears in the job log ──────────────────────────
python -c "
import json
d = json.load(open('vlm_prompt_runner/results/phase1_qwen3_vl_8b.json'))
print()
print('Best prompt:', d['best_prompt'])
print()
print('All prompt accuracies:')
for stem, stats in sorted(d['prompts'].items(), key=lambda x: -x[1]['accuracy']):
    marker = ' ← best' if stem == d['best_prompt'] else ''
    print(f'  {stem:<35} {stats[\"correct\"]}/{stats[\"total\"]}  ({stats[\"accuracy\"]:.1%}){marker}')
"

echo ""
echo "Phase 1 complete: $(date)"
echo "Results: $PROJECT_ROOT/vlm_prompt_runner/results/phase1_qwen3_vl_8b.json"
