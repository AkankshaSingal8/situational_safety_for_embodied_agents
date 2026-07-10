#!/bin/bash -l
#SBATCH --job-name=base_og10_L2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=08:00:00
#SBATCH --mem=63000M
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/base_og10_L2_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/base_og10_L2_%j.err

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$REPO/.worktrees/fol-safety-filter"

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p "$WT/baseline_og10_L2" "$WT/baseline_og10_L2_video"

echo "=== Baseline Evaluation (no filter) ===" && echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"

command -v module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null
module load anaconda3
conda activate openvla_libero_merged

export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache
export PYTHONPATH="$WT:$REPO/vlm_pipeline:$REPO/openvla-oft:$REPO/SafeLIBERO/safelibero:${PYTHONPATH:-}"

python "$WT/vlm_pipeline/run_safelibero_fol_openvla_eval.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name safelibero_spatial \
    --safety_level II \
    --num_trials_per_task 50 \
    --results_output_dir "$WT/baseline_og10_L2" \
    --video_output_dir "$WT/baseline_og10_L2_video" \
    --center_crop True \
    --seed 7 \
    --run_id_note "baseline-og10-L2"

echo "=== Baseline complete: $(date) ==="
