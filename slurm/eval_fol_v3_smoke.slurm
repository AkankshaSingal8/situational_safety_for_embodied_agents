#!/bin/bash
#SBATCH --job-name=fol_v3_smoke
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=5
#SBATCH --partition=GPU-shared --gres=gpu:v100-32:1
#SBATCH --time=01:00:00 --mem=63000M
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v3_smoke_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/fol_v3_smoke_%j.err

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$REPO/.worktrees/fol-safety-filter"
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs "$WT/fol_v3_smoke"
echo "=== FOL v3 SMOKE TEST safelibero_goal Level I n=2 ===" && echo "Start: $(date)"
module load anaconda3 && conda activate openvla_libero_merged
export MUJOCO_GL=egl HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONPATH="$WT/v3:$WT:$REPO/vlm_pipeline:$REPO/openvla-oft:$REPO/SafeLIBERO/safelibero:${PYTHONPATH:-}"
python "$WT/vlm_pipeline/run_safelibero_fol_openvla_eval.py" \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
    --task_suite_name safelibero_goal --safety_level I \
    --num_trials_per_task 2 --use_fol_filter True --fol_level 1 \
    --results_output_dir "$WT/fol_v3_smoke" --video_output_dir "$WT/fol_v3_smoke_video" \
    --center_crop True --seed 7 --run_id_note "fol-v3-smoke"
echo "=== Done: $(date) ==="
