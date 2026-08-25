#!/bin/bash
#SBATCH --job-name=fol_object_L1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=5
#SBATCH --partition=GPU-shared --gres=gpu:v100-32:1
#SBATCH --time=08:00:00 --mem=63000M
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/fol_object_L1_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/fol_object_L1_%j.err

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$REPO/.worktrees/fol-safety-filter"
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs "$WT/fol_object_L1"
echo "=== FOL L1 safelibero_object Level I ===" && echo "Start: $(date)"
module load anaconda3 && conda activate openvla_libero_merged
export MUJOCO_GL=egl HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONPATH="$WT:$REPO/vlm_pipeline:$REPO/openvla-oft:$REPO/SafeLIBERO/safelibero:${PYTHONPATH:-}"
python "$WT/vlm_pipeline/run_safelibero_fol_openvla_eval.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-object --task_suite_name safelibero_object --safety_level I \
    --num_trials_per_task 10 --use_fol_filter True --fol_level 1 \
    --results_output_dir "$WT/fol_object_L1" --video_output_dir "$WT/fol_object_L1_video" \
    --center_crop True --seed 7 --run_id_note "fol-level1-v1-final"
echo "=== Done: $(date) ==="
