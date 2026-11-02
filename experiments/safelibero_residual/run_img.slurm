#!/bin/bash
#SBATCH -J slres_img
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 05:30:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Iteration 1 (selection only, no test-set runs):
#  - recollect on the cached training scenes WITH images: oracle-driven + c0-residual-driven (DAgger states)
#  - train feature residual and image residual (text / no-text) on the same data, layouts 20-24 held out
#  - offline validation fit on held-out layouts (fit.json) is the primary selection signal;
#    closed-loop gate on official episodes 20-24 (n=40, same layouts as the held-out rows) is secondary
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safelibero_residual
PY=/ocean/projects/cis250185p/asingal/envs/openvla_libero_merged/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=$PWD
CK=runs/base/ema_last.pt; G=runs/gate; SC=data/scenes_train.npz
D1=data/residual_i1_or; D2=data/residual_i1_dag; D=$D1,$D2
HO="--scenes $SC --holdout_layouts 20-24"
[ -f $D1/agent_0.npy ] || $PY collect_residual.py --ckpt $CK --scenes $SC --out $D1 --workers 5 --save_images --seed 2
[ -f $D2/agent_0.npy ] || $PY collect_residual.py --ckpt $CK --scenes $SC --out $D2 --workers 5 --save_images --seed 1 \
    --residual runs/res_text_c0/residual.pt
for V in text notext; do
  [ -f runs/res_${V}_i1/residual.pt ] || $PY train_residual.py --data $D --out runs/res_${V}_i1 --variant $V --steps 20000 $HO 2>&1 | grep -v -i warn
done
for V in text_img notext_img; do
  [ -f runs/res_${V}_i1/residual.pt ] || $PY train_residual.py --data $D --out runs/res_${V}_i1 --variant $V --steps 25000 --bs 128 $HO 2>&1 | grep -v -i warn
done
for V in text notext text_img notext_img; do echo "$V $(cat runs/res_${V}_i1/fit.json)"; done
EV="$PY eval_sl.py --ckpt $CK --workers 5 --eps 20-24"
$EV --out $G/res_i1_text_true --arm residual --residual runs/res_text_i1/residual.pt --text_mode true
$EV --out $G/res_i1_img_true --arm residual --residual runs/res_text_img_i1/residual.pt --text_mode true
$EV --out $G/res_i1_img_wrong --arm residual --residual runs/res_text_img_i1/residual.pt --text_mode wrong
$EV --out $G/res_i1_notextimg --arm residual --residual runs/res_notext_img_i1/residual.pt --text_mode none
$PY compare.py $G/base $G/oracle_c0 $G/res_c0_true_g1 $G/res_i1_text_true $G/res_i1_img_true $G/res_i1_img_wrong $G/res_i1_notextimg \
    --eps 20-24 --json $G/i1.json || true
echo ALL_DONE
