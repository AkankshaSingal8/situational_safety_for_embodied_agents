#!/bin/bash
# Train the GR00T residual expert (residual/train_groot_residual.py) in the groot_srv_u24 env (Ubuntu 24.04 sandbox).
#   DATA=/data/asingal2/oopsieverse/residual/data_r0 OUT=/data/asingal2/oopsieverse/residual/runs/r0 STEPS=4000 \
#   sbatch -J ov_train -p pleiades -x pleiades-1-3,pleiades-1-10 --time=12:00:00 -c 8 --mem=96G --gres=gpu:1 \
#     -o /data/asingal2/slurm_logs/%x_%j.out train_groot_residual.slurm
set -uo pipefail
DATA=${DATA:?set DATA}
OUT=${OUT:?set OUT}
STEPS=${STEPS:-4000}
BS=${BS:-16}
EXTRA=${EXTRA:-}
R=/home/asingal2/safety/wt-oopsieverse-eval/oopsieverse_eval/residual
mkdir -p $OUT
echo "host $(hostname) data=$DATA out=$OUT steps=$STEPS bs=$BS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/data/asingal2/envs/apptainer/bin/apptainer exec --nv --bind /data --bind /home/asingal2 \
  --env HF_HOME=/data/asingal2/hf_cache --env PYTHONUNBUFFERED=1 /data/asingal2/oopsieverse/b1k_sandbox \
  /data/asingal2/envs/groot_srv_u24/bin/python $R/train_groot_residual.py --data $DATA --out $OUT \
  --steps $STEPS --bs $BS $EXTRA
RC=$?; echo "EXIT=$RC"; exit $RC
