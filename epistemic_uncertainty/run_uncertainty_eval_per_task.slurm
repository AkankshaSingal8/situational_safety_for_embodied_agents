#!/bin/bash
#SBATCH --job-name=unc_eval_task
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
#SBATCH --output=experiments/logs/uncertainty_eval_task%a_%j.out
#SBATCH --error=experiments/logs/uncertainty_eval_task%a_%j.err
#SBATCH --array=0-3

# ── User-configurable parameters ─────────────────────────────────────────────
CHECKPOINT="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
SUITE="safelibero_spatial"      # safelibero_spatial | safelibero_object | safelibero_goal | safelibero_long
SAFETY_LEVEL="I"                # I | II
NUM_TRIALS=50
MC_SAMPLES=10
UNCERTAINTY_METHODS="mc_dropout entropy"   # space-separated subset of:
                                                   #   mc_dropout entropy density conformal deep_ensemble
OUTPUT_DIR="results/uncertainty_eval"

# Task ID comes from the SLURM array index (0-3).
# Override with --array=N or pass TASK_ID=N via sbatch --export.
TASK_ID="${TASK_ID:-${SLURM_ARRAY_TASK_ID}}"

# OOD detection (only used if "density" is in UNCERTAINTY_METHODS)
OOD_BACKBONE="dinov2"           # dinov2 | vla
OOD_CAL_DIR="vlm_inputs/safelibero_spatial/level_I/task_1/episode_00"

# Conformal prediction (only used if "conformal" is in UNCERTAINTY_METHODS)
CONFORMAL_ALPHA=0.1
# CONFORMAL_CAL_JSON="results/uncertainty_eval/conformal_cal_scores.json"

# Deep ensemble (only used if "deep_ensemble" is in UNCERTAINTY_METHODS)
# ENSEMBLE_CKPTS="/path/to/ckpt1 /path/to/ckpt2 /path/to/ckpt3"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "=========================================="
echo "SLURM Job ID : $SLURM_JOB_ID"
echo "Array Task   : $TASK_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Suite        : $SUITE  Level: $SAFETY_LEVEL"
echo "Methods      : $UNCERTAINTY_METHODS"
echo "Start time   : $(date)"
echo "=========================================="

# ── Environment setup ─────────────────────────────────────────────────────────
module load anaconda3
conda activate openvla_libero_merged

export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

# Suppress noisy MuJoCo/EGL warnings
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=${SLURM_LOCALID:-0}

WORKDIR="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents"
cd "$WORKDIR"

mkdir -p experiments/logs
mkdir -p "$OUTPUT_DIR"

# ── Build optional argument flags ─────────────────────────────────────────────
EXTRA_ARGS=""

if [[ "${UNCERTAINTY_METHODS}" == *"density"* ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --ood-backbone $OOD_BACKBONE"
    if [[ -n "${OOD_CAL_DIR:-}" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --ood-calibration-dir $OOD_CAL_DIR"
    fi
fi

if [[ "${UNCERTAINTY_METHODS}" == *"conformal"* ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --conformal-alpha $CONFORMAL_ALPHA"
    if [[ -n "${CONFORMAL_CAL_JSON:-}" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --conformal-calibration-json $CONFORMAL_CAL_JSON"
    fi
fi

if [[ "${UNCERTAINTY_METHODS}" == *"deep_ensemble"* ]]; then
    if [[ -n "${ENSEMBLE_CKPTS:-}" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --ensemble-checkpoints $ENSEMBLE_CKPTS"
    fi
fi

# ── Run evaluation ────────────────────────────────────────────────────────────
python epistemic_uncertainty/run_safelibero_uncertainty_eval.py \
    --pretrained-checkpoint "$CHECKPOINT" \
    --task-suite-name "$SUITE" \
    --safety-level "$SAFETY_LEVEL" \
    --num-trials-per-task "$NUM_TRIALS" \
    --uncertainty-methods $UNCERTAINTY_METHODS \
    --mc-samples "$MC_SAMPLES" \
    --output-dir "$OUTPUT_DIR" \
    --task-ids "$TASK_ID" \
    $EXTRA_ARGS

echo "=========================================="
echo "End time : $(date)"
echo "Task ID  : $TASK_ID"
echo "Results  : $OUTPUT_DIR"
echo "=========================================="
