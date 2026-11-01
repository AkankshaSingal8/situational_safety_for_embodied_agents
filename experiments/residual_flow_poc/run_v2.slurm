#!/bin/bash
#SBATCH -J rflow_v2
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 03:00:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# v2 pipeline.   sbatch --export=ALL,MODE=smoke run_v2.slurm   (then MODE=full)
#  smoke: regenerate data_v2 with the simulated control filter, v1 regression checks,
#         300-step training, control check on 100+100 scenes, tiny eval, analysis, 1 clip/group
#  full:  30k-step training of main / notext / shuffled / main_ab on the existing data_v2,
#         eval on train and test scenes, analysis, render the artifact clips
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/residual_flow_poc
PY=/ocean/projects/cis250185p/asingal/envs/flow_policy/bin/python
export OMP_NUM_THREADS=1
MODE=${MODE:-smoke}
SEED=${SEED:-0}          # training seed; full runs with SEED>0 go to runs_v2_s$SEED and skip rendering
if [ "$MODE" = smoke ]; then STEPS=300; RUNS=runs_v2_smoke; else STEPS=30000; RUNS=runs_v2; fi
TAG=$MODE
if [ "$MODE" = full ] && [ "$SEED" != 0 ]; then RUNS=runs_v2_s$SEED; TAG=full_s$SEED; fi
mkdir -p $RUNS logs

train_all() {   # train the four variants concurrently; fail the job if any of them fails
  local pids=()
  rm -f $RUNS/*/ema.pt
  for V in main notext shuffled main_ab; do
    $PY train.py --variant $V --data_dir data_v2 --out $RUNS/$V --steps $STEPS --seed $SEED > logs/v2_${TAG}_train_$V.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  tail -n 1 logs/v2_${TAG}_train_*.log
}

if [ "$MODE" = smoke ]; then
  # 1. v2 data with the control enforced in simulation
  $PY gen_data.py --profile v2 --sim_filter --workers 5 > logs/v2_gen.log 2>&1
  cat logs/v2_gen.log
  # 2. v1 regression checks: training stream and physics unchanged
  SCR=$RUNS/v1check; mkdir -p $SCR
  for V in main notext shuffled; do
    $PY train.py --variant $V --data_dir data --out $SCR/$V --steps 200 2>&1 | grep "'step': 0," | sed "s/^/v1 $V /"
  done
  echo "v1 expected step-0 loss: main/notext 1.2370390892028809, shuffled 1.243929147720337"
  $PY - <<'PYEOF'
import json, numpy as np
from src.sim import PandaScene
TE = np.load("data/test.npz", allow_pickle=True)
W = np.load("runs/eval/waypoints.npz", allow_pickle=True)
R = [json.loads(l) for _, l in zip(range(40), open("runs/eval/rollouts.jsonl"))]
bad = 0
for k in range(0, 40, 4):
    m = json.loads(W["meta"][k]); i = m["scene"]
    r = PandaScene(TE["scenes"][i]).rollout(TE["base"][i].astype(np.float64) + W["residual"][k].astype(np.float64))
    bad += any(r[q] != R[k][q] for q in ("viol_A", "viol_B", "success", "max_tilt_deg", "final_dist"))
print("v1 physics replay mismatches:", bad, "of 10")
PYEOF
  # 3. short training
  train_all
  # 4. control check on all 100 scenes of each split (the first 600 jobs are the references)
  for S in test train; do
    $PY evaluate_v2.py --split $S --runs $RUNS --out $RUNS/ctrl_$S --limit 600 > logs/v2_smoke_ctrl_$S.log 2>&1
    $PY - $RUNS/ctrl_$S <<'PYEOF'
import json, sys, collections
by = collections.defaultdict(list)
for l in open(sys.argv[1] + "/rollouts.jsonl"):
    r = json.loads(l); by[r["cond"]].append(r)
for c, rs in by.items():
    m = lambda k: sum(r[k] for r in rs) / len(rs)
    print(sys.argv[1], c, len(rs), "violA %.2f violB %.2f succ %.2f maxtrack %.3f" % (m("viol_A"), m("viol_B"), m("success"), max(r["max_track_err"] for r in rs)))
PYEOF
  done
  # 5. every condition on 5 scenes per split exercises the whole pipeline
  for S in test train; do
    $PY evaluate_v2.py --split $S --runs $RUNS --out $RUNS/eval_$S --n_scenes 5 > logs/v2_smoke_eval_$S.log 2>&1
  done
  $PY analyze_v2.py --evals $RUNS/eval_train $RUNS/eval_test --out $RUNS > logs/v2_smoke_analyze.log 2>&1
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl $PY render_v2.py --evals $RUNS/eval_train $RUNS/eval_test --out $RUNS/artifact --n 1 > logs/v2_smoke_render.log 2>&1
else
  train_all
  for S in test train; do
    $PY evaluate_v2.py --split $S --runs $RUNS --out $RUNS/eval_$S --workers 5 > logs/v2_${TAG}_eval_$S.log 2>&1
    tail -n 1 logs/v2_${TAG}_eval_$S.log
  done
  $PY analyze_v2.py --evals $RUNS/eval_train $RUNS/eval_test --out $RUNS > logs/v2_${TAG}_analyze.log 2>&1
  if [ "$SEED" = 0 ]; then
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl $PY render_v2.py --evals $RUNS/eval_train $RUNS/eval_test --out $RUNS/artifact --n 5 > logs/v2_full_render.log 2>&1
  fi
fi
[ -f logs/v2_${TAG}_render.log ] && tail -n 3 logs/v2_${TAG}_render.log || true
echo ALL_DONE
