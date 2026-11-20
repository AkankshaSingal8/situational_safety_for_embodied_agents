#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Scale-up residual pipeline for one suite and one base (see SCALEUP.md). Every stage is skipped when its
# output exists, so the job can be resubmitted after a timeout.
#   sbatch -J res_object --export=ALL,SUITE=safelibero_object,STAGE=select run_residual_scale.slurm
#   BASE=pi05 additionally needs a pi0.5 server on PI_PORT (see run_pi05_server.sh)
# STAGE=select : P2 gate (episodes 0-19) -> scenes -> round-0 oracle data -> c0 feature residual ->
#                DAgger data -> i1b grid (seed 0) with offline validation fit -> closed-loop validation
#                gate on official episodes 20-24.  No test-set episode is touched.
# STAGE=seeds  : the selected config (CFG=steps_lr, written after selection) with seeds 1 and 2.
# STAGE=test   : ONE run on official episodes 25-49 for every arm and seed, plus two-obstacle scenes.
set -euo pipefail
: "${SUITE:?}" "${STAGE:?}"
S=${SUITE#safelibero_}
BASE=${BASE:-flow}
PI_PORT=${PI_PORT:-8000}
R=/home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual/pleiades_run.sh
CK=runs_scale/$S/base/ema_last.pt
OUT=runs_scale/$S/$BASE          # metrics (small, committed)
D=data_scale/$S/$BASE            # rollouts and images (large, /data)
SC=data_scale/$S/scenes_train.npz; S2=data_scale/$S/scenes_test2.npz
mkdir -p $OUT $D
if [ "$BASE" = pi05 ]; then
  RES=256; EX=5
  # pi0.5 LIBERO server on this job's GPU (needs an Ada node: submit with --exclude=pleiades-1-3,pleiades-1-10)
  PI_PORT=$((8000 + RANDOM % 1000))
  cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
  source ./run_pi05_server.sh pi05_libero $PI_PORT
  trap 'kill $PI_SERVER_PID 2>/dev/null || true' EXIT
else RES=128; EX=8; fi
COMMON="--suite $SUITE --ckpt $CK --base $BASE --pi_port $PI_PORT --env_seed 0 --settle 20 --exec_steps $EX"
EV="python eval_sl.py $COMMON --res $RES --workers 6"
COL="python collect_residual.py $COMMON --res $RES --workers 6 --save_images"
HO="--scenes $SC --holdout_layouts 20-24"
CMP="python compare.py"

if [ "$STAGE" = select ]; then
  # P2 gate: base and oracle on official episodes 0-19 (layouts that training scenes also start from)
  $R $EV --out $OUT/gate/base --eps 0-19
  $R $EV --out $OUT/gate/oracle --arm oracle --text_mode true --eps 0-19
  $R $CMP $OUT/gate/base $OUT/gate/oracle --eps 0-19 --json $OUT/gate/gate.json || true
  # scenes are shared by both bases of a suite
  [ -f $SC ] || $R python scenes.py --suite $SUITE --out $SC --layouts 0-24 --per_layout 6 --seed 0
  [ -f $S2 ] || $R python scenes.py --suite $SUITE --out $S2 --layouts 25-49 --per_layout 1 --n_obs 2 --seed 1
  # round 0: oracle drives; c0 feature residual (text) for the DAgger round
  [ -f $D/r0/agent_0.npy ] || $R $COL --scenes $SC --out $D/r0 --seed 2
  [ -f $OUT/c0/residual.pt ] || $R python train_residual.py --data $D/r0 --out $OUT/c0 --variant text --steps 20000 $HO
  [ -f $D/dag/agent_0.npy ] || $R $COL --scenes $SC --out $D/dag --seed 1 --residual $OUT/c0/residual.pt
  # i1b grid, training seed 0 (SCALEUP.md section 3): steps x lr, text and no-text
  for STEPS in 25000 50000; do for LR in 3e-4 1e-4; do
    C=s${STEPS}_lr${LR}
    for V in text_img notext_img; do
      [ -f $OUT/$C/${V}_seed0/residual.pt ] || $R python train_residual.py --data $D/r0,$D/dag --out $OUT/$C/${V}_seed0 \
          --variant $V --steps $STEPS --lr $LR --bs 128 --seed 0 --base_ckpt $CK $HO
    done
    # closed-loop validation gate: official episodes 20-24 (layouts held out of residual training)
    $R $EV --out $OUT/val/${C}_true --arm residual --residual $OUT/$C/text_img_seed0/residual.pt --text_mode true --eps 20-24
  done; done
  $R $EV --out $OUT/val/base --eps 20-24
  # selection is pooled over the four suites: run select_config.py once every suite has finished this stage
fi

if [ "$STAGE" = seeds ]; then
  C=$(python3 -c "import json; print(json.load(open('$OUT/selected.json'))['config'])")
  read STEPS LR <<<"$(python3 -c "c='$C'; print(c.split('_')[0][1:], c.split('_lr')[1])")"
  for SEED in 1 2; do for V in text_img notext_img; do
    [ -f $OUT/$C/${V}_seed$SEED/residual.pt ] || $R python train_residual.py --data $D/r0,$D/dag --out $OUT/$C/${V}_seed$SEED \
        --variant $V --steps $STEPS --lr $LR --bs 128 --seed $SEED --base_ckpt $CK $HO
  done; done
fi

if [ "$STAGE" = test ]; then
  # Runs once. Every arm, every seed, all reported (SCALEUP.md section 3).
  C=$(python3 -c "import json; print(json.load(open('$OUT/selected.json'))['config'])")
  T=$OUT/test
  $R $EV --out $T/base --eps 25-49
  $R $EV --out $T/oracle --arm oracle --text_mode true --eps 25-49
  for SEED in 0 1 2; do
    RT=$OUT/$C/text_img_seed$SEED/residual.pt; RN=$OUT/$C/notext_img_seed$SEED/residual.pt
    for M in true held wrong none; do
      $R $EV --out $T/s$SEED/res_$M --arm residual --residual $RT --text_mode $M --eps 25-49
    done
    $R $EV --out $T/s$SEED/notext --arm residual --residual $RN --text_mode none --eps 25-49
    $R $EV --scenes $S2 --out $T/s$SEED/two_res_true --arm residual --residual $RT --text_mode true
    $R $EV --scenes $S2 --out $T/s$SEED/two_notext --arm residual --residual $RN --text_mode none
  done
  $R $EV --scenes $S2 --out $T/two_base
  $R $EV --scenes $S2 --out $T/two_oracle --arm oracle --text_mode true
fi
echo ALL_DONE
