#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Text controls for the pi0.5 LIBERO + i1b residual (Spatial, config s50000_lr3e-4 = pooled-val pick, seed 0).
# Question: is the gain driven by the safety sentence?  No test-set episode (25-49) is touched.
#   primary   : official episodes 20-24 (layouts held out of residual training); correct-sentence and base arms
#               already exist in runs_scale/spatial/pi05/val/{s50000_lr3e-4_true,base}.
#   secondary : official episodes 0-19 (training layouts, obstacles re-sampled) for power; base = gate/base.
#   sbatch -J textctl_pi05 experiments/safelibero_residual/run_textctl_pi05.slurm
set -euo pipefail
S=spatial; SUITE=safelibero_spatial; C=s50000_lr3e-4
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
R=./pleiades_run.sh
PI_PORT=$((8000 + RANDOM % 1000))
source ./run_pi05_server.sh pi05_libero $PI_PORT
trap 'kill $PI_SERVER_PID 2>/dev/null || true' EXIT
CK=runs_scale/$S/base/ema_last.pt
EV="python eval_sl.py --suite $SUITE --ckpt $CK --base pi05 --pi_port $PI_PORT --env_seed 0 --settle 20 --exec_steps 5 --res 256 --workers 6"
O=runs_scale/$S/pi05
RT=$O/$C/text_img_seed0/residual.pt; RN=$O/$C/notext_img_seed0/residual.pt
T=$O/textctl
for M in wrong none held; do
  $R $EV --out $T/val_res_$M --arm residual --residual $RT --text_mode $M --eps 20-24
done
$R $EV --out $T/val_notext --arm residual --residual $RN --text_mode none --eps 20-24
for M in true wrong none; do
  $R $EV --out $T/tr_res_$M --arm residual --residual $RT --text_mode $M --eps 0-19
done
$R $EV --out $T/tr_notext --arm residual --residual $RN --text_mode none --eps 0-19
echo ALL_DONE
