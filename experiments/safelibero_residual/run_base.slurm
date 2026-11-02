#!/bin/bash
#SBATCH -J slres_base
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 04:00:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Base image flow policy.  sbatch --export=ALL,MODE=smoke run_base.slurm   (then MODE=full)
#  smoke: pack 3 demos/task, 300 training steps, 4 SafeLIBERO episodes + 2 obstacle-free ones
#  full : pack all demos, STEPS training steps, official SafeLIBERO-Spatial I+II (400 episodes),
#         and the same scenes with the obstacle removed for episodes 0-9 (80 episodes)
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safelibero_residual
PY=/ocean/projects/cis250185p/asingal/envs/openvla_libero_merged/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=$PWD TORCH_HOME=/ocean/projects/cis250185p/asingal/hf_cache/torch
SRC=/ocean/projects/cis250185p/asingal/data_libero/libero_spatial
MODE=${MODE:-smoke}
STEPS=${STEPS:-50000}
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ "$MODE" = smoke ]; then
  DATA=/ocean/projects/cis250185p/asingal/data_libero/packed_smoke; RUN=runs/base_smoke
  $PY prep_data.py --src $SRC --out $DATA --max_demos 3
  $PY train_base.py --data $DATA --out $RUN --steps 300 --save_every 300
  $PY eval_sl.py --ckpt $RUN/ema_last.pt --out $RUN/eval_official --levels I --tasks 0,3 --eps 0-1 --workers 2
  $PY eval_sl.py --ckpt $RUN/ema_last.pt --out $RUN/eval_noobs --levels II --tasks 1 --eps 0-1 --workers 2 --no_obstacle
else
  DATA=/ocean/projects/cis250185p/asingal/data_libero/packed_spatial; RUN=runs/base
  [ -f $DATA/meta.json ] || $PY prep_data.py --src $SRC --out $DATA
  [ -f $RUN/ema_${STEPS}.pt ] || $PY train_base.py --data $DATA --out $RUN --steps $STEPS
  $PY eval_sl.py --ckpt $RUN/ema_last.pt --out $RUN/eval_official --workers 5
  $PY eval_sl.py --ckpt $RUN/ema_last.pt --out $RUN/eval_noobs --eps 0-9 --workers 5 --no_obstacle
fi
echo ALL_DONE
