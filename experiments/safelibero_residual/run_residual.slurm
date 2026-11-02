#!/bin/bash
#SBATCH -J slres_res
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 08:00:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Residual stage.  sbatch --export=ALL,ORACLE_CFG='{"mode": "side"}',ORACLE_TAG=c1 run_residual.slurm
#  1. gate: the chosen oracle and base on official episodes 0-24 (extends the tuning runs)
#  2. collect oracle data on the generated training scenes (layouts 0-24)
#  3. train the text residual and the no-text control
#  4. held-out official episodes 25-49: base, oracle, residual x {true, held, wrong, none}, no-text
#  5. held-out two-obstacle scenes (layouts 25-49): base, oracle, residual true, no-text
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safelibero_residual
PY=/ocean/projects/cis250185p/asingal/envs/openvla_libero_merged/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=$PWD
CK=runs/base/ema_last.pt; G=runs/gate; T=runs/test; mkdir -p $T
OC=${ORACLE_CFG:-'{}'}; TAG=${ORACLE_TAG:-c0}
RD=data/residual_$TAG; RT=runs/res_text_$TAG; RN=runs/res_notext_$TAG
EV="$PY eval_sl.py --ckpt $CK --workers 5"

# 1. gate on episodes 0-24 (resumes the 0-9 tuning runs)
$EV --out $G/base --eps 0-24
$EV --out $G/oracle_$TAG --arm oracle --text_mode true --oracle_cfg "$OC" --eps 0-24
$PY compare.py $G/base $G/oracle_$TAG --eps 0-24 --json $G/gate_$TAG.json || true
$EV --out runs/base/eval_noobs_v2 --eps 0-9 --no_obstacle
[ -f data/scenes_train.npz ] || $PY scenes.py --out data/scenes_train.npz --layouts 0-24 --per_layout 6 --seed 0
[ -f data/scenes_test2.npz ] || $PY scenes.py --out data/scenes_test2.npz --layouts 25-49 --per_layout 1 --n_obs 2 --seed 1

# 2-3. data and training
[ -f $RD/part_0.npz ] || $PY collect_residual.py --ckpt $CK --scenes data/scenes_train.npz --out $RD --oracle_cfg "$OC" --workers 5
[ -f $RT/residual.pt ] || $PY train_residual.py --data $RD --out $RT --variant text
[ -f $RN/residual.pt ] || $PY train_residual.py --data $RD --out $RN --variant notext

# 4. held-out official episodes
$EV --out $T/base --eps 25-49
$EV --out $T/oracle_$TAG --arm oracle --text_mode true --oracle_cfg "$OC" --eps 25-49
for M in true held wrong none; do
  $EV --out $T/res_${TAG}_$M --arm residual --residual $RT/residual.pt --text_mode $M --eps 25-49
done
$EV --out $T/notext_$TAG --arm residual --residual $RN/residual.pt --text_mode none --eps 25-49
$PY compare.py $T/base $T/oracle_$TAG $T/res_${TAG}_true $T/res_${TAG}_held $T/res_${TAG}_wrong $T/res_${TAG}_none $T/notext_$TAG \
  --eps 25-49 --json $T/official_$TAG.json || true

# 5. two-obstacle held-out scenes: 'true' names the first obstacle
S2="--scenes data/scenes_test2.npz"
$EV $S2 --out $T/two_base
$EV $S2 --out $T/two_oracle_$TAG --arm oracle --text_mode true --oracle_cfg "$OC"
$EV $S2 --out $T/two_res_${TAG}_true --arm residual --residual $RT/residual.pt --text_mode true
$EV $S2 --out $T/two_notext_$TAG --arm residual --residual $RN/residual.pt --text_mode none
$PY compare.py $T/two_base $T/two_oracle_$TAG $T/two_res_${TAG}_true $T/two_notext_$TAG --json $T/two_$TAG.json || true
echo ALL_DONE
