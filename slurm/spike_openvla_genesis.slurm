#!/bin/bash
#SBATCH --job-name=spike_openvla_genesis
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:45:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
ENV_PATH=/ocean/projects/cis250185p/asingal/envs/genesis_env
HF_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache
OUTPUT_DIR=$REPO/genesis_bakeoff/spike_openvla_e2e_output

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p "$OUTPUT_DIR"

module load cuda/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

export HF_HOME=$HF_CACHE
export HF_HUB_CACHE=$HF_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=$HF_CACHE
export MUJOCO_GL=egl
export PYTHONPATH=$REPO/vlm_pipeline:$REPO/openvla-oft:$REPO/SafeLIBERO/safelibero:${PYTHONPATH:-}

echo "=========================================="
echo " Dependency versions in genesis_env"
echo "=========================================="
python -c "
import torch, transformers, peft, timm, genesis
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('timm:', timm.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"

echo "=========================================="
echo " Running OpenVLA-OFT x Genesis E2E spike"
echo "=========================================="

python "$REPO/genesis_bakeoff/spike_openvla_e2e.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --steps 15 \
    --output_dir "$OUTPUT_DIR"

SPIKE_EXIT=$?

echo ""
echo "=========================================="
echo " Spike result (exit code: $SPIKE_EXIT)"
echo "=========================================="

exit $SPIKE_EXIT
