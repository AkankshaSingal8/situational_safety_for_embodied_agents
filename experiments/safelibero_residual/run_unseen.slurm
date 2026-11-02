#!/bin/bash
#SBATCH -J slres_unseen
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 03:00:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Unseen-object test: image residuals (i1b recipe) trained WITHOUT milk carton and red mug (no scene containing
# them, no sentence naming them) and without layouts 20-24; evaluated on held-out official episodes 25-49,
# reported separately for the unseen obstacle types and the seen ones.
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safelibero_residual
PY=/ocean/projects/cis250185p/asingal/envs/openvla_libero_merged/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=$PWD
CK=runs/base/ema_last.pt; T=runs/test; D=data/residual_i1_or,data/residual_i1_dag
UO=milk_obstacle_1,red_coffee_mug_obstacle_1
HO="--scenes data/scenes_train.npz --holdout_layouts 20-24 --holdout_objects $UO"
for V in text_img notext_img; do
  [ -f runs/res_${V}_uo/residual.pt ] || $PY train_residual.py --data $D --out runs/res_${V}_uo --variant $V --steps 25000 --bs 128 $HO 2>&1 | grep -v -i warn
done
EV="$PY eval_sl.py --ckpt $CK --workers 5 --eps 25-49"
$EV --out $T/uo_true --arm residual --residual runs/res_text_img_uo/residual.pt --text_mode true
$EV --out $T/uo_none --arm residual --residual runs/res_text_img_uo/residual.pt --text_mode none
$EV --out $T/uo_notext --arm residual --residual runs/res_notext_img_uo/residual.pt --text_mode none
echo "== UNSEEN obstacle types"
$PY compare.py $T/base $T/uo_true $T/uo_none $T/uo_notext --eps 25-49 --obstacles $UO --json $T/uo_unseen.json || true
echo "== SEEN obstacle types"
$PY compare.py $T/base $T/uo_true $T/uo_none $T/uo_notext --eps 25-49 --obstacles "!$UO" --json $T/uo_seen.json || true
echo ALL_DONE
