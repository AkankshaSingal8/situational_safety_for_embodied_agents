#!/bin/bash
#SBATCH -J slres_img2
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 02:30:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# i1b: diagnose the collapsed image residual (BatchNorm hypothesis), retrain with frozen encoder BN,
# gate on episodes 20-24 (selection only).
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safelibero_residual
PY=/ocean/projects/cis250185p/asingal/envs/openvla_libero_merged/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=$PWD
CK=runs/base/ema_last.pt; G=runs/gate; D=data/residual_i1_or,data/residual_i1_dag
HO="--scenes data/scenes_train.npz --holdout_layouts 20-24"
echo "DIAG bn_batch on the old text_img checkpoint:"
$PY train_residual.py --data $D --out runs/res_text_img_i1 --variant text_img --bn_free --eval_only runs/res_text_img_i1/residual.pt --bn_batch $HO 2>&1 | grep "fit "
for V in text_img notext_img; do
  [ -f runs/res_${V}_i1b/residual.pt ] || $PY train_residual.py --data $D --out runs/res_${V}_i1b --variant $V --steps 25000 --bs 128 $HO 2>&1 | grep -v -i warn
done
for V in text_img notext_img; do echo "$V i1b $(cat runs/res_${V}_i1b/fit.json)"; done
EV="$PY eval_sl.py --ckpt $CK --workers 5 --eps 20-24"
$EV --out $G/res_i1b_img_true --arm residual --residual runs/res_text_img_i1b/residual.pt --text_mode true
$EV --out $G/res_i1b_img_wrong --arm residual --residual runs/res_text_img_i1b/residual.pt --text_mode wrong
$EV --out $G/res_i1b_notextimg --arm residual --residual runs/res_notext_img_i1b/residual.pt --text_mode none
$PY compare.py $G/base $G/oracle_c0 $G/res_c0_true_g1 $G/res_i1_text_true $G/res_i1b_img_true $G/res_i1b_img_wrong $G/res_i1b_notextimg \
    --eps 20-24 --json $G/i1b.json || true
echo ALL_DONE
