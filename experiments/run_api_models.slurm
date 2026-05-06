#!/bin/bash
#SBATCH --job-name=vlm_api
#SBATCH --output=experiments/logs/api_%A_%a.out
#SBATCH --error=experiments/logs/api_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7GB
#SBATCH --time=06:00:00
#SBATCH --partition=RM-shared
#SBATCH --account=cis250185p
#SBATCH --qos=low
#SBATCH --array=0-6

# One array task per API model — no GPU required.
# Submit: sbatch experiments/run_api_models.slurm

set -euo pipefail

MODELS=(
    claude-sonnet-4-6
    claude-opus-4-7
    gpt-4.1-mini
    gpt-4.1
    gpt-5.5
    gemini-2.5-flash
    gemini-2.5-pro
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

PROJECT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
cd "$PROJECT"

module load anaconda3
conda activate vlm_api

mkdir -p experiments/logs

for TASK in 0 1 2 3; do
    echo "[$(date)] Model=$MODEL  Task=$TASK"
    python -m vlm_prompt_runner.run_multi_model_experiment \
        --prompts-dir prompts/obstacle_id \
        --models "$MODEL" \
        --suite safelibero_spatial \
        --level I \
        --task "$TASK" \
        --results-out "vlm_prompt_runner/results/${MODEL}/task_${TASK}.json"
done

echo "[$(date)] Done — model=$MODEL"
