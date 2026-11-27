#!/bin/bash
set -uo pipefail
P=/home/asingal2/safety/wt-oopsieverse-eval/oopsieverse_eval/pleiades
R=/data/asingal2/oopsieverse/residual
while squeue -h -j 67799 | grep -q .; do sleep 120; done
grep -h "TRAIN_DONE\|Traceback" /data/asingal2/slurm_logs/ov_train_r0_67799.out | tail -2
[ -f $R/runs/r0/last.pt ] || { echo "no checkpoint"; exit 1; }
cd $P
for t in open_single_door turn_on_microwave counter_to_microwave turn_on_stove; do
  STOP_ON_SUCCESS=1 SEED=700 NUM_TRIALS=15 TASK=$t OUT=$R/val_base/$t sbatch --parsable -J ov_vbase_$t -p pleiades -x pleiades-1-3,pleiades-1-10 --time=06:00:00 -c 8 --mem=64G --gres=gpu:1 -o /data/asingal2/slurm_logs/%x_%j.out eval_groot_stock_damagesim.slurm
  RES_CKPT=$R/runs/r0/last.pt STOP_ON_SUCCESS=1 SEED=700 NUM_TRIALS=15 TASK=$t OUT=$R/val_res_r0/$t sbatch --parsable -J ov_vres_$t -p pleiades -x pleiades-1-3,pleiades-1-10 --time=08:00:00 -c 8 --mem=96G --gres=gpu:1 -o /data/asingal2/slurm_logs/%x_%j.out eval_groot_stock_damagesim.slurm
done
echo VAL_SUBMITTED
