#!/bin/bash
#SBATCH --job-name=goalII_fol
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Goal L2 no-GT strict-SR fix, attempt 2 (identity swap): F3 2-view refuted
# the localization hypothesis (job 42877126: t1 ident_correct 0.14, strict
# 39.5 = board). The symbolic scorer misidentifies the t1 hazard regardless
# of position accuracy -> swap identity source to the FOL compiler (E3: FOL
# >= symbolic on both spatial cells). Board config otherwise unchanged.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/fgd_goalII_fol"
PRIORS="$WT/results_tables/vlm_hazard_priors.json"
test -s "$PRIORS"
PORT=${PORT:-8286}
N=${N:?export N explicitly}
CPP="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.10 \
  --policy.config=pi05_libero --policy.dir="$AEGIS/checkpoints/pi05_libero" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do
  python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
  sleep 5
done
kill -0 "$SERVER_PID"

conda activate openvla_libero_merged
export MUJOCO_GL=egl
PYTHONPATH="$CPP" python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_goal --safety_level II \
  --num_trials_per_task "$N" --host 127.0.0.1 --port "$PORT" --corridor \
  --obstacle_pos_source percep --obstacle_id_source fol \
  --entity_pos_source percep \
  --hazard_prior_source vlm --vlm_prior_json "$PRIORS" \
  --results_output_dir "$OUT/results"
echo "GOALII_FOL_COMPLETE"
