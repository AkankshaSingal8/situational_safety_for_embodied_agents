#!/bin/bash
#SBATCH --job-name=vlm_bench
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w005,w006,w007,w008,w009
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# VLM go/no-go ablation: local Qwen2.5-VL 3B+7B on slots S1 (property head)
# and S2 (whole-scene referee). Offline only — no eval interaction.
set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen
WT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/flow-guidance-tier-gt
cd "$WT"
for B in qwen3b qwen7b; do
  for S in s1 s2; do
    python vlm_pipeline/vlm_slot_bench.py --backend "$B" --slot "$S" \
      --n_votes 5 --limit_per_task 5 \
      --out "results_tables/vlm_bench_${S}_${B}.jsonl"
  done
done
echo "VLM_BENCH_QWEN_COMPLETE"
