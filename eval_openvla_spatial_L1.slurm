#!/bin/bash
#SBATCH --job-name=openvla_spatial_L1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

# ── Environment ────────────────────────────────────────────────────────────────
module load anaconda3
conda activate openvla_libero_merged

export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export PYTHONPATH=/ocean/projects/cis250185p/asingal/openvla-oft:$PYTHONPATH
export PYTHONPATH=/ocean/projects/cis250185p/asingal/SafeLIBERO/safelibero:$PYTHONPATH

cd /ocean/projects/cis250185p/asingal

# ── Run evaluation ─────────────────────────────────────────────────────────────
python run_safelibero_openvla_oft_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 50 \
    --video_output_dir /ocean/projects/cis250185p/asingal/openvla_video \
    --results_output_dir /ocean/projects/cis250185p/asingal/openvla_benchmark
