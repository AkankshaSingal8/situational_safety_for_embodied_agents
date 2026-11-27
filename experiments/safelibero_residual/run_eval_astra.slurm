#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# GPT-6 Astra arms (ASTRA.md) on SafeLIBERO, same settings as ex2's run_eval_pg.slurm.
#   ARM=astra_base : Astra is the policy. No model servers; the GPU is only for EGL rendering.
#   ARM=astra_res  : pi0.5 (pi05_libero server) proposes, Astra returns the xyz residual.
#   sbatch -J ab_spatial_40-49 --export=ALL,S=spatial,ARM=astra_base,EPS=40-49 run_eval_astra.slurm
#   SCENES=scenes_newlayout | SCENES=../../data_gso/spatial/scenes_newobj25  (as in run_eval_pg.slurm)
#   SCENE_IDS=lo-hi splits a scene file across jobs. WORKERS (default 16) = parallel episodes = parallel API calls.
#   EXTRA adds eval_sl flags (e.g. "--levels I --tasks 0"); OUT overrides the output folder (smoke tests).
#   ASTRA_EFFORT sets reasoning_effort ('' = API default).
# Out: runs_v2/<S>/<test|TAG>/<ARM>/true/<eps<EPS>|all>; Astra call log in <out>/astra_calls/.
set -euo pipefail
: "${S:?}" "${ARM:?}"
EPS=${EPS:-40-49}
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
set -a; . $HOME/.envfile; set +a
R=./pleiades_run.sh
PI=""
if [ "$ARM" = astra_res ] || [ "$ARM" = fable_res ]; then
  PI_PORT=$((8000 + RANDOM % 1000))
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
  source ./run_pi05_server.sh pi05_libero $PI_PORT
  trap 'kill $PI_SERVER_PID 2>/dev/null || true' EXIT
  PI="--pi_port $PI_PORT"
fi
TAG=test; SUB=eps$EPS; SC=""
if [ -n "${SCENES:-}" ]; then TAG=$(basename $SCENES); SUB=all; SC="--scenes data_v2/$S/$SCENES.npz ${SCENE_IDS:+--scene_ids $SCENE_IDS}"; fi
$R python eval_sl.py --suite safelibero_$S --ckpt runs_scale/$S/base/ema_last.pt --base pi05 $PI --env_seed 0 --settle 20 \
  --exec_steps 5 --res 256 --workers ${WORKERS:-16} --eps $EPS $SC --arm $ARM --text_mode true \
  --astra_effort "${ASTRA_EFFORT:-}" ${EXTRA:-} --out ${OUT:-runs_v2/$S/$TAG/$ARM/true/$SUB}
echo ALL_DONE
