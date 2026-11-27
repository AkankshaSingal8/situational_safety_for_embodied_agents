#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-08:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# DAgger round 2, step 1: new TRAINING obstacle placements on layouts 0-39 for SafeLIBERO obstacles.
#   sbatch -J d2gen_spatial --export=ALL,S=spatial run_dag2.slurm   -> data_v2/<S>/scenes_train_d.npz
# Every placement is >= 3 cm from all train / val placements (newpos_scenes.REFS) and from every Table 3 candidate
# and test placement (--extra_refs), so Table 3 positions stay unseen. Different seed from Table 3 (51).
# Step 2 (rollouts, ex8 driving, shield labels): run_dagger_ex.slurm MODE=v2 SCENES=data_v2/<S>/scenes_train_d.npz
set -euo pipefail
: "${S:?}"
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
./pleiades_run.sh python newpos_scenes.py gen --suite safelibero_$S --seed 61 --per_layout ${PER:-3} \
    --extra_refs scenes_newpos_cand.npz,scenes_newpos.npz --out data_v2/$S/scenes_train_d.npz
echo ALL_DONE
