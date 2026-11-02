#!/bin/bash
#SBATCH -J slres_pipe
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 00:45:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Residual pipeline smoke on a 100-step base: checks, scene generation, oracle arm,
# data collection, residual training, residual arms. Every stage runs on a handful of items.
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safelibero_residual
PY=/ocean/projects/cis250185p/asingal/envs/openvla_libero_merged/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=$PWD TORCH_HOME=/ocean/projects/cis250185p/asingal/hf_cache/torch
SRC=/ocean/projects/cis250185p/asingal/data_libero/libero_spatial
DATA=/ocean/projects/cis250185p/asingal/data_libero/packed_smoke_pipe
R=runs/pipe_smoke; rm -rf $R; mkdir -p $R
$PY checks.py
[ -f $DATA/meta.json ] || $PY prep_data.py --src $SRC --out $DATA --max_demos 3
$PY train_base.py --data $DATA --out $R/base --steps 100 --save_every 100
CK=$R/base/ema_last.pt
time $PY scenes.py --out $R/scenes.npz --levels I,II --tasks 2 --layouts 0-1 --per_layout 2
time $PY eval_sl.py --ckpt $CK --out $R/eval_oracle --arm oracle --text_mode true --levels I --tasks 2 --eps 0-1 --workers 2
time $PY eval_sl.py --ckpt $CK --out $R/eval_scn_base --scenes $R/scenes.npz --workers 2
time $PY collect_residual.py --ckpt $CK --scenes $R/scenes.npz --out $R/resdata --workers 2
$PY train_residual.py --data $R/resdata --out $R/res_text --variant text --steps 300
$PY train_residual.py --data $R/resdata --out $R/res_notext --variant notext --steps 300
for M in true wrong held none; do
  $PY eval_sl.py --ckpt $CK --out $R/eval_res_$M --arm residual --residual $R/res_text/residual.pt --text_mode $M --levels II --tasks 2 --eps 0 --workers 1
done
$PY eval_sl.py --ckpt $CK --out $R/eval_res_notext --arm residual --residual $R/res_notext/residual.pt --text_mode none --levels II --tasks 2 --eps 0 --workers 1
echo ALL_DONE
