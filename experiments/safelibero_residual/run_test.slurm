#!/bin/bash
#SBATCH -J slres_test
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 03:30:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# ONE held-out test run of a variant already selected on the gate/validation data.
#   sbatch --export=ALL,TAG=i1img,RT=runs/res_text_img_i1/residual.pt,RN=runs/res_notext_img_i1/residual.pt run_test.slurm
# Base and oracle test runs are reused from runs/test (same episodes, deterministic env seeding).
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safelibero_residual
PY=/ocean/projects/cis250185p/asingal/envs/openvla_libero_merged/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=$PWD
CK=runs/base/ema_last.pt; T=runs/test
EV="$PY eval_sl.py --ckpt $CK --workers 5"
for M in true held wrong none; do
  $EV --out $T/res_${TAG}_$M --arm residual --residual $RT --text_mode $M --eps 25-49
done
$EV --out $T/notext_$TAG --arm residual --residual $RN --text_mode none --eps 25-49
$PY compare.py $T/base $T/oracle_c0 $T/res_c0_true $T/res_${TAG}_true $T/res_${TAG}_held $T/res_${TAG}_wrong $T/res_${TAG}_none $T/notext_$TAG \
  --eps 25-49 --json $T/official_$TAG.json || true
$PY compare.py $T/res_${TAG}_wrong $T/res_${TAG}_true --eps 25-49 || true
$PY compare.py $T/notext_$TAG $T/res_${TAG}_true --eps 25-49 || true
S2="--scenes data/scenes_test2.npz"
$EV $S2 --out $T/two_res_${TAG}_true --arm residual --residual $RT --text_mode true
$EV $S2 --out $T/two_notext_$TAG --arm residual --residual $RN --text_mode none
$PY compare.py $T/two_base $T/two_oracle_c0 $T/two_res_c0_true $T/two_res_${TAG}_true $T/two_notext_$TAG --json $T/two_$TAG.json || true
echo ALL_DONE
