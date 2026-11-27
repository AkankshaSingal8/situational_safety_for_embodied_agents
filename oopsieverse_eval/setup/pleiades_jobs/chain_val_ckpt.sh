#!/bin/bash
# usage: chain_val_ckpt.sh <train_jobid> <run_name>   -> after training, val residual runs/<run>/last.pt on seeds 700-714
set -uo pipefail
J=$1; RUN=$2; SUF=${3:-}   # RES_N / GATE_THR from the environment reach the eval job (sbatch exports env)
P=/home/asingal2/safety/wt-oopsieverse-eval/oopsieverse_eval/pleiades
R=/data/asingal2/oopsieverse/residual
[ "$J" = none ] || while squeue -h -j $J 2>/dev/null | grep -q .; do sleep 120; done
grep -h "TRAIN_DONE\|Traceback\|Error" /data/asingal2/slurm_logs/*_$J.out | tail -3
[ -f $R/runs/$RUN/last.pt ] || { echo "no checkpoint"; exit 1; }
cd $P
for t in open_single_door turn_on_microwave counter_to_microwave turn_on_stove; do
  RES_CKPT=$R/runs/$RUN/last.pt STOP_ON_SUCCESS=1 SEED=700 NUM_TRIALS=15 TASK=$t OUT=$R/val_res_$RUN$SUF/$t sbatch --parsable -J ov_vres_${RUN}${SUF}_$t -p pleiades -x pleiades-1-3,pleiades-1-10 --time=08:00:00 -c 8 --mem=96G --gres=gpu:1 -o /data/asingal2/slurm_logs/%x_%j.out eval_groot_stock_damagesim.slurm
done
echo VAL_SUBMITTED
