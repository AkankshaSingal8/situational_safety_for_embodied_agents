#!/bin/bash
#SBATCH -J slres_scale
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 08:00:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Object-scaling study with the extended SafeLIBERO-Spatial scenes (19 obstacle types, extend_bddl.py).
# Stages (each skips work already on disk; run in order, one job per stage fits the 8 h cap):
#   STAGE=scenes   generate extended train / seen-test / unseen-test scenes        (CPU-bound, ~2-4 h)
#   STAGE=collect  oracle-driven collection with images on the train scenes        (~1.5 h)
#   STAGE=train    per pool size K: image residual with safety text + instruction text, and the no-text control
#   STAGE=eval     per K: official eps 25-49 and the extended test scenes; correct sentence / no sentence / no-text
#   sbatch --export=ALL,STAGE=scenes run_scale.slurm
#
# Split (nothing evaluated is trained on):
#   layouts   train 0-19, validation 20-24, test 25-49 (official episodes 25-49 + extended test scenes)
#   objects   HELD5 never appear in any training scene or sentence; they form the unseen-object test
#   pools     K6 = the original six; K10 = K6 + 4 new; K14 = K6 + 8 new (nested); every K trains on
#             MAXC chunks so the curve measures object diversity, not data size
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safelibero_residual
PY=/ocean/projects/cis250185p/asingal/envs/openvla_libero_merged/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=$PWD
CK=runs/base/ema_last.pt; X=runs/scale; mkdir -p $X
O=_obstacle_1
ORIG6="moka_pot$O,white_storage_box$O,milk$O,wine_bottle$O,red_coffee_mug$O,yellow_book$O"
NEW_A="orange_juice$O,butter$O,basket$O,simple_rack$O"
NEW_B="bbq_sauce$O,chocolate_pudding$O,new_salad_dressing$O,basin_faucet_base$O"
HELD5="ketchup$O,cream_cheese$O,popcorn$O,chefmate_8_frypan$O,alphabet_soup$O"
K14="$ORIG6,$NEW_A,$NEW_B"
declare -A EXCL=( [6]="$NEW_A,$NEW_B,$HELD5" [10]="$NEW_B,$HELD5" [14]="$HELD5" )
MAXC=${MAXC:-12000}
SC=data/scenes_x_train.npz; TS=data/scenes_x_test_seen.npz; TU=data/scenes_x_test_unseen.npz; D=data/residual_x

case "${STAGE:?set STAGE}" in
scenes)
  $PY checks.py 2>&1 | grep -E "^ok|PASSED|Error|assert"
  $PY extend_bddl.py
  [ -f $SC ] || $PY scenes.py --extended --types "$K14" --out $SC --layouts 0-24 --per_layout 10 --seed 10
  [ -f $TS ] || $PY scenes.py --extended --types "$K14" --out $TS --layouts 25-49 --per_layout 1 --n_obs 1 --seed 11
  [ -f $TU ] || $PY scenes.py --extended --types "$HELD5" --out $TU --layouts 25-49 --per_layout 1 --n_obs 1 --seed 12
  ;;
collect)
  [ -f $D/agent_0.npy ] || $PY collect_residual.py --ckpt $CK --scenes $SC --out $D --workers 5 --save_images --seed 3
  ;;
train)
  for K in ${KS:-6 10 14}; do
    for V in text_img notext_img; do
      [ -f $X/res_${V}_K$K/residual.pt ] || $PY train_residual.py --data $D --scenes $SC --out $X/res_${V}_K$K --variant $V \
          --steps 25000 --bs 128 --instr --holdout_layouts 20-24 --holdout_objects "${EXCL[$K]}" --max_train_chunks $MAXC \
          2>&1 | grep -v -i warn
    done
  done
  ;;
eval)
  EV="$PY eval_sl.py --ckpt $CK --workers 5"
  # once: base and oracle on the extended test scenes (official 25-49 base/oracle runs already exist in runs/test)
  for S in seen unseen; do
    F=data/scenes_x_test_$S.npz
    $EV --scenes $F --out $X/${S}_base
    $EV --scenes $F --out $X/${S}_oracle --arm oracle --text_mode true
  done
  for K in ${KS:-6 10 14}; do
    RT=$X/res_text_img_K$K/residual.pt; RN=$X/res_notext_img_K$K/residual.pt
    for M in true none; do
      $EV --eps 25-49 --out $X/off_K${K}_$M --arm residual --residual $RT --text_mode $M
      for S in seen unseen; do
        $EV --scenes data/scenes_x_test_$S.npz --out $X/${S}_K${K}_$M --arm residual --residual $RT --text_mode $M
      done
    done
    $EV --eps 25-49 --out $X/off_K${K}_notext --arm residual --residual $RN --text_mode none
    for S in seen unseen; do
      $EV --scenes data/scenes_x_test_$S.npz --out $X/${S}_K${K}_notext --arm residual --residual $RN --text_mode none
    done
    $PY compare.py runs/test/base $X/off_K${K}_none $X/off_K${K}_true $X/off_K${K}_notext --eps 25-49 --json $X/off_K$K.json || true
    for S in seen unseen; do
      $PY compare.py $X/${S}_base $X/${S}_K${K}_none $X/${S}_K${K}_true $X/${S}_K${K}_notext $X/${S}_oracle --json $X/${S}_K$K.json || true
    done
  done
  ;;
esac
echo ALL_DONE
