#!/bin/bash
#SBATCH --job-name=baseline_fol_spatial_L1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=08:00:00
#SBATCH --mem=63000M
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/baseline_fol_L1_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/baseline_fol_L1_%j.err

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$REPO/.worktrees/fol-safety-filter"

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p "$WT/baseline_benchmark" "$WT/baseline_video"

echo "=== Baseline Evaluation (no filter) ===" && echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"

module load anaconda3
conda activate openvla_libero_merged

export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache
export PYTHONPATH="$WT:$REPO/vlm_pipeline:$REPO/openvla-oft:$REPO/SafeLIBERO/safelibero:${PYTHONPATH:-}"

python "$WT/vlm_pipeline/run_safelibero_fol_openvla_eval.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 10 \
    --results_output_dir "$WT/baseline_benchmark" \
    --video_output_dir "$WT/baseline_video" \
    --center_crop True \
    --seed 7 \
    --run_id_note "baseline-no-filter"

echo "=== Baseline complete: $(date) ==="
