#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# v2 residual: cache frozen SigLIP features (STAGE=cache) or train the joint residual (STAGE=train).
#   sbatch -J v2_cache --export=ALL,STAGE=cache run_train_v2.slurm
#   sbatch -J v2_train --export=ALL,STAGE=train,OUT=runs_v2/train/joint_r0 run_train_v2.slurm
set -euo pipefail
: "${STAGE:?}"
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
if [ "$STAGE" = cache ]; then
  ./pleiades_run.sh python cache_feats_v2.py --suites ${SUITES:-spatial,object,goal,long} --rounds ${ROUNDS:-r0} \
      --out ${FEATS:-data_v2/feats_siglip}
else
  : "${OUT:?}"
  ./pleiades_run.sh python train_v2.py --feats ${FEATS:-data_v2/feats_siglip} --suites ${SUITES:-spatial,object,goal,long} \
      --out $OUT --steps ${STEPS:-50000} --lr ${LR:-3e-4} --nz_frac ${NZ:-0}
fi
echo ALL_DONE
