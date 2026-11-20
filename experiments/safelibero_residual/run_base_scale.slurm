#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Scale-up P0/P1: flow base for one suite (recipe unchanged from the Spatial study: 50k steps, bs 128,
# lr 1e-4, seed 0), then base evaluation under the scale-up protocol (env seed 0, 20 settle steps):
#   all 50 official episodes x 4 tasks x 2 levels, and the obstacle-removed scenes for episodes 0-9.
#   sbatch -J base_object --export=ALL,SUITE=safelibero_object run_base_scale.slurm
# For spatial it also reruns official episodes 0-24 under the PSC protocol (settle 10) for the P0 check.
set -euo pipefail
: "${SUITE:?}"
S=${SUITE#safelibero_}
declare -A DEMO=([spatial]=libero_spatial [object]=libero_object [goal]=libero_goal [long]=libero_10)
SRC=/data/asingal2/data_libero/${DEMO[$S]}
DATA=/data/asingal2/data_libero/packed_$S
RUN=runs_scale/$S/base
R=/home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual/pleiades_run.sh
mkdir -p /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual/$RUN
[ -f $DATA/meta.json ] || $R python prep_data.py --src $SRC --out $DATA
[ -f $RUN/ema_50000.pt ] || $R python train_base.py --data $DATA --out $RUN --steps 50000 --seed 0
EV="python eval_sl.py --suite $SUITE --ckpt $RUN/ema_last.pt --workers 6 --env_seed 0 --settle 20"
$R $EV --out $RUN/eval_official
$R $EV --out $RUN/eval_noobs --eps 0-9 --no_obstacle
if [ "$S" = spatial ]; then
  $R python eval_sl.py --suite $SUITE --ckpt $RUN/ema_last.pt --workers 6 --env_seed 0 --settle 10 --eps 0-24 --out $RUN/eval_p0_psc_protocol
  $R python compare.py runs/gate/base $RUN/eval_p0_psc_protocol --eps 0-24 || true
fi
echo ALL_DONE
