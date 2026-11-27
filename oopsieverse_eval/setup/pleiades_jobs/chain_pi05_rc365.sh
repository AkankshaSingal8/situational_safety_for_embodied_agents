#!/bin/bash
# wait for download -> check norm stats -> smoke -> full paper-protocol run (no early stop)
set -uo pipefail
L=/data/asingal2/slurm_logs; P=/home/asingal2/safety/wt-oopsieverse-eval/oopsieverse_eval/pleiades
R=/data/asingal2/oopsieverse/results/robocasa_papermatch


python3 -c "
import json;s=json.load(open('/data/asingal2/openpi_cache/robocasa365/pi05_pretrain_human300/multitask_learning/75000/assets/norm_stats.json'))['norm_stats']
print({k:[kk for kk,vv in v.items() if vv is not None] for k,v in s.items()})"
cd $P
J=$(MAX_STEPS=400 NUM_TRIALS=2 CKPT=pi05_robocasa365 TASKS=turn_on_stove OUT=$R/smoke_pi05_rc365 sbatch --parsable -J ov_rc365p05_smoke -p pleiades -x pleiades-1-3,pleiades-1-10 --time=01:30:00 -c 8 --mem=48G --gres=gpu:1 -o $L/%x_%j.out eval_robocasa_papermatch.slurm)
while squeue -h -j $J | grep -q .; do sleep 60; done
grep -h "SUMMARY\|EXIT=\|server died" $L/ov_rc365p05_smoke_$J.out
grep -q "EXIT=0" $L/ov_rc365p05_smoke_$J.out || { echo SMOKE FAILED; tail -30 $R/smoke_pi05_rc365/server_$J.log; exit 1; }
for t in open_single_door turn_on_microwave counter_to_microwave turn_on_stove pick_egg turn_on_faucet open_drawer close_drawer place_plate serve_pastry prepare_coffee prepare_breakfast dishes_to_sink nav_lift_bowl wipe_counter; do
  case $t in serve_pastry|prepare_coffee|prepare_breakfast|dishes_to_sink|nav_lift_bowl|wipe_counter) S=800; T=08:00:00;; *) S=400; T=05:00:00;; esac
  MAX_STEPS=$S CKPT=pi05_robocasa365 TASKS=$t OUT=$R/pi05_robocasa365_full/$t sbatch --parsable -J ov_rc365p05full_$t -p pleiades -x pleiades-1-3,pleiades-1-10 --time=$T -c 8 --mem=48G --gres=gpu:1 -o $L/%x_%j.out eval_robocasa_papermatch.slurm
done
echo FULL_SUBMITTED
