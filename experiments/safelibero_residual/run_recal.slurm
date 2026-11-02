#!/bin/bash
#SBATCH -J slres_recal
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 01:15:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Repair BN stats of the i1 image residuals, then gate them on official episodes 20-24 (no test-set runs).
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safelibero_residual
PY=/ocean/projects/cis250185p/asingal/envs/openvla_libero_merged/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=$PWD
CK=runs/base/ema_last.pt; G=runs/gate; D=data/residual_i1_or,data/residual_i1_dag
for V in text_img notext_img; do
  $PY recal_bn.py --ckpt runs/res_${V}_i1/residual.pt --data $D --scenes data/scenes_train.npz --holdout_layouts 20-24 2>&1 | grep -v -i warn
done
EV="$PY eval_sl.py --ckpt $CK --workers 5 --eps 20-24"
rm -rf $G/res_i1_img_true $G/res_i1_img_wrong $G/res_i1_notextimg
$EV --out $G/res_i1_img_true --arm residual --residual runs/res_text_img_i1/residual_bn.pt --text_mode true
$EV --out $G/res_i1_img_wrong --arm residual --residual runs/res_text_img_i1/residual_bn.pt --text_mode wrong
$EV --out $G/res_i1_notextimg --arm residual --residual runs/res_notext_img_i1/residual_bn.pt --text_mode none
$PY compare.py $G/base $G/oracle_c0 $G/res_c0_true_g1 $G/res_i1_text_true $G/res_i1_img_true $G/res_i1_img_wrong $G/res_i1_notextimg \
    --eps 20-24 --json $G/i1_bn.json || true
echo ALL_DONE
