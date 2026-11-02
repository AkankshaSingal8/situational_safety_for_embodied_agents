#!/bin/bash
#SBATCH -J slres_guide
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 02:00:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Diagnose the c0 residual and select a text-guidance weight on the GATE episodes 0-24 (not the test set).
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safelibero_residual
PY=/ocean/projects/cis250185p/asingal/envs/openvla_libero_merged/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=$PWD
CK=runs/base/ema_last.pt; G=runs/gate; RT=runs/res_text_c0/residual.pt
$PY diag_fit.py data/residual_c0 $RT runs/res_notext_c0/residual.pt 2>&1 | grep -v -i warn
for W in 1 2 3; do
  $PY eval_sl.py --ckpt $CK --workers 5 --out $G/res_c0_true_g$W --arm residual --residual $RT --text_mode true --guide $W --eps 0-24
done
$PY compare.py $G/base $G/oracle_c0 $G/res_c0_true_g1 $G/res_c0_true_g2 $G/res_c0_true_g3 --eps 0-24 --json $G/guide_c0.json || true
echo ALL_DONE
