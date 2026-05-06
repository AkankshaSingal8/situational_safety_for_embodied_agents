#!/bin/bash
#SBATCH --job-name=vlm_qwen
#SBATCH --output=experiments/logs/qwen_%A_%a.out
#SBATCH --error=experiments/logs/qwen_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=16GB
#SBATCH --time=08:00:00
#SBATCH --partition=GPU-shared
#SBATCH --account=cis250185p
#SBATCH --qos=gpu
#SBATCH --array=0-3

# One array task per Qwen model variant — GPU required.
# Submit: sbatch experiments/run_qwen_models.slurm

set -euo pipefail

MODELS=(
    qwen2.5-vl-3b
    qwen2.5-vl-7b
    qwen2-vl-7b
    qwen3-vl-8b
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

PROJECT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
cd "$PROJECT"

module load anaconda3
conda activate qwen

mkdir -p experiments/logs

for TASK in 0 1 2 3; do
    echo "[$(date)] Model=$MODEL  Task=$TASK"
    python -m vlm_prompt_runner.run_multi_model_experiment \
        --prompts-dir prompts/obstacle_id \
        --models "$MODEL" \
        --suite safelibero_spatial \
        --level I \
        --task "$TASK" \
        --load-in-4bit \
        --results-out "vlm_prompt_runner/results/${MODEL}/task_${TASK}.json"
done

echo "[$(date)] Done — model=$MODEL"
