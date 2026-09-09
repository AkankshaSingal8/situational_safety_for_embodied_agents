#!/bin/bash
#SBATCH --job-name=fol_openvla_spatial_L2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=08:00:00
#SBATCH --mem=63000M
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/fol_openvla_L2_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/fol_openvla_L2_%j.err

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$REPO/.worktrees/fol-safety-filter"

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p "$WT/fol_benchmark_L2" "$WT/fol_video_L2"

echo "=== FOL Safety Filter Evaluation (Spatial Level II) ===" && echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"

module load anaconda3
conda activate openvla_libero_merged

export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache
export PYTHONPATH="$WT:$REPO/vlm_pipeline:$REPO/openvla-oft:$REPO/SafeLIBERO/safelibero:${PYTHONPATH:-}"

python "$WT/vlm_pipeline/run_safelibero_fol_openvla_eval.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name safelibero_spatial \
    --safety_level II \
    --num_trials_per_task 10 \
    --use_fol_filter True \
    --fol_level 3 \
    --results_output_dir "$WT/fol_benchmark_L2" \
    --video_output_dir "$WT/fol_video_L2" \
    --center_crop True \
    --seed 7 \
    --run_id_note "fol-level3-safety-levelII"

echo "=== FOL Level II eval complete: $(date) ==="
