#!/bin/bash
#SBATCH --job-name=smpoc_run
#SBATCH --account=cis240028p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
set -uo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
conda activate openvla_libero_merged
export MUJOCO_GL=egl
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safety_memory_poc
: "${PAIRS:=20}"
: "${OUT:=runs/libero_main}"
python run_campaign.py --sim libero --suite safelibero_spatial --task_id 0 \
  --pairs "$PAIRS" --out "$OUT" \
  --marked_source akita_black_bowl_1_pos \
  --clean_source  akita_black_bowl_2_pos \
  --rte           plate_1_pos \
  --wash          glazed_rim_porcelain_ramekin_1_pos \
  --d_contact 0.06 --d_viol 0.08 --margin 0.13 \
  --max_steps 320 --exposure_steps 260 --action_scale 0.0113 --overshoot 0.6
python analyze.py --runs "$OUT" --out "$OUT/report.md"
