#!/bin/bash
# Parameterized OpenVLA-OFT evaluation on SafeLIBERO.
# Called by submit_all_evals.sh with --export=ALL,SUITE=...,LEVEL=...
#
# Usage (direct):
#   sbatch --export=ALL,SUITE=safelibero_spatial,LEVEL=I slurm/eval_openvla_safelibero.slurm
#
# GPU: v100-32 — openvla_libero_merged env was built/tested against V100 CUDA drivers.
#SBATCH --job-name=openvla_eval
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

# ── Validate parameters ────────────────────────────────────────────────────────
: "${SUITE:?ERROR: SUITE env var not set. Use --export=ALL,SUITE=safelibero_spatial,...}"
: "${LEVEL:?ERROR: LEVEL env var not set. Use --export=ALL,LEVEL=I or II}"

echo "=== OpenVLA-OFT Eval: SUITE=$SUITE, LEVEL=$LEVEL ==="

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs
mkdir -p "$REPO/openvla_benchmark"
mkdir -p "$REPO/openvla_video"

# ── Checkpoint mapping (task-specific models) ──────────────────────────────────
case "$SUITE" in
    safelibero_spatial) CKPT="moojink/openvla-7b-oft-finetuned-libero-spatial" ;;
    safelibero_object)  CKPT="moojink/openvla-7b-oft-finetuned-libero-object"  ;;
    safelibero_goal)    CKPT="moojink/openvla-7b-oft-finetuned-libero-goal"    ;;
    safelibero_long)    CKPT="moojink/openvla-7b-oft-finetuned-libero-10"      ;;
    *)                  echo "ERROR: Unknown SUITE=$SUITE"; exit 1 ;;
esac
echo "Checkpoint: $CKPT"

# ── Conda (module load is required on Bridges2 for this env) ──────────────────
module load anaconda3
conda activate openvla_libero_merged

export MUJOCO_GL=egl
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export TRANSFORMERS_CACHE=/ocean/projects/cis250185p/asingal/.hf_cache
export PYTHONPATH="$REPO/vlm_pipeline:$REPO/openvla-oft:$REPO/SafeLIBERO/safelibero:${PYTHONPATH:-}"

# ── Run evaluation ─────────────────────────────────────────────────────────────
python "$REPO/vlm_pipeline/run_safelibero_openvla_oft_eval.py" \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name "$SUITE" \
    --safety_level "$LEVEL" \
    --num_trials_per_task 50 \
    --results_output_dir "$REPO/openvla_benchmark" \
    --video_output_dir "$REPO/openvla_video" \
    --center_crop True \
    --seed 7

echo "=== Done: SUITE=$SUITE, LEVEL=$LEVEL ==="
