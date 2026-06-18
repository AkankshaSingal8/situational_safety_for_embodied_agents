#!/bin/bash
# OpenVLA-OFT smoke test: 5 episodes on spatial Level I, all tasks.
# Results go to openvla_benchmark_smoke/ (separate from main benchmark dir).
# Uses v100-32 GPU — the openvla_libero_merged env was built against V100 CUDA drivers.
#SBATCH --job-name=openvla_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=02:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

echo "=== OpenVLA Smoke Test: spatial Level I, 5 episodes/task ==="

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p "$REPO/openvla_benchmark_smoke"

# ── Conda (use module load, exactly like the working April eval script) ─────
module load anaconda3
conda activate openvla_libero_merged

export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache
export PYTHONPATH="$REPO/vlm_pipeline:$REPO/openvla-oft:$REPO/SafeLIBERO/safelibero:${PYTHONPATH:-}"

python "$REPO/vlm_pipeline/run_safelibero_openvla_oft_eval.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 5 \
    --results_output_dir "$REPO/openvla_benchmark_smoke" \
    --center_crop True \
    --seed 7

echo "=== Smoke test done ==="
echo "Check results: $REPO/openvla_benchmark_smoke/safelibero_spatial/I/"
