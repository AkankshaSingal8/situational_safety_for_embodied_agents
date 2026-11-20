#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00
#SBATCH --exclude=pleiades-1-3
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Residual flow head over a frozen backbone (train_pg.py).
#   selection: sbatch -J sel_pg --export=ALL,BB=pg,HOLD=1,OUT=runs_v2/pg/sel_pg run_train_pg.slurm
#   final    : SUITES=spatial,object,goal,long,goal_off sbatch -J fin_pg --export=ALL,BB=pg,OUT=runs_v2/pg/final_pg run_train_pg.slurm
#   (commas inside --export split variables, so pass SUITES through the environment)
set -euo pipefail
: "${BB:?}" "${OUT:?}"
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
./pleiades_run.sh python train_pg.py --backbone $BB --suites ${SUITES:-spatial,object,goal,long} --out $OUT \
    --steps ${STEPS:-40000} --lr ${LR:-3e-4} --nz_frac ${NZ:-0.3} ${HOLD:+--holdout}
echo ALL_DONE
