#!/bin/bash
#SBATCH -J slres_dag
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 04:00:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# DAgger round: the previous text residual drives on the cached TRAINING scenes, the oracle labels
# the visited states; retrain on the aggregate; select on GATE episodes 0-24 (no test-set runs here).
#   sbatch --export=ALL,PREV=c0,NEW=dag1,GUIDE=2,DATA=data/residual_c0 run_dagger.slurm
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safelibero_residual
PY=/ocean/projects/cis250185p/asingal/envs/openvla_libero_merged/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=$PWD
CK=runs/base/ema_last.pt; G=runs/gate
PREV=${PREV:-c0}; NEW=${NEW:-dag1}; W=${GUIDE:-1}; DATA=${DATA:-data/residual_c0}
RD=data/residual_$NEW; RT=runs/res_text_$NEW; RN=runs/res_notext_$NEW
[ -f $RD/part_0.npz ] || $PY collect_residual.py --ckpt $CK --scenes data/scenes_train.npz --out $RD --workers 5 \
    --residual runs/res_text_$PREV/residual.pt --guide $W --seed 1
[ -f $RT/residual.pt ] || $PY train_residual.py --data $DATA,$RD --out $RT --variant text --steps 30000
[ -f $RN/residual.pt ] || $PY train_residual.py --data $DATA,$RD --out $RN --variant notext --steps 30000
$PY diag_fit.py $RD $RT/residual.pt $RN/residual.pt 2>&1 | grep -v -i warn || true
EV="$PY eval_sl.py --ckpt $CK --workers 5 --eps 0-24"
$EV --out $G/res_${NEW}_true_g$W --arm residual --residual $RT/residual.pt --text_mode true --guide $W
$EV --out $G/res_${NEW}_wrong_g$W --arm residual --residual $RT/residual.pt --text_mode wrong --guide $W
$EV --out $G/notext_$NEW --arm residual --residual $RN/residual.pt --text_mode none
$PY compare.py $G/base $G/oracle_c0 $G/res_c0_true_g$W $G/res_${NEW}_true_g$W $G/res_${NEW}_wrong_g$W $G/notext_$NEW \
    --eps 0-24 --json $G/dagger_$NEW.json || true
echo ALL_DONE
