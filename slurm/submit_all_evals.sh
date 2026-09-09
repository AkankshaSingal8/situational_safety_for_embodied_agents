#!/bin/bash
# Submit all OpenVLA and Pi0.5 SafeLIBERO evaluation jobs.
#
# Usage:
#   bash slurm/submit_all_evals.sh              # smoke test first, then full sweep
#   bash slurm/submit_all_evals.sh --skip-smoke  # skip smoke, submit Pi0.5 immediately
#   bash slurm/submit_all_evals.sh --dry-run     # print sbatch commands without submitting
#   bash slurm/submit_all_evals.sh --openvla-only
#   bash slurm/submit_all_evals.sh --pi05-only
#
# After all jobs finish, generate tables:
#   python scripts/aggregate_results.py

set -euo pipefail
cd "$(dirname "$0")/.."   # run from repo root

SLURM_DIR=slurm
SKIP_SMOKE=false
DRY_RUN=false
OPENVLA_ONLY=false
PI05_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --skip-smoke)    SKIP_SMOKE=true ;;
        --dry-run)       DRY_RUN=true ;;
        --openvla-only)  OPENVLA_ONLY=true ;;
        --pi05-only)     PI05_ONLY=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

SUITES=(safelibero_spatial safelibero_object safelibero_goal safelibero_long)
LEVELS=(I II)

sbatch_cmd() {
    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch $*" >&2
        echo "99999999"   # fake job ID for dry-run dependency chaining
    else
        sbatch "$@" | awk '{print $NF}'
    fi
}

declare -A JOB_IDS
echo ""
echo "========================================================"
echo " SafeLIBERO Benchmark — Job Submission"
echo "========================================================"

# ── Smoke test ────────────────────────────────────────────────────────────────
SMOKE_DEP=""
if ! $SKIP_SMOKE && ! $OPENVLA_ONLY; then
    echo ""
    echo "--- Pi0.5 Smoke Test ---"
    SMOKE_JOB_ID=$(sbatch_cmd \
        --job-name="pi05_smoke" \
        "$SLURM_DIR/smoke_pi05_safelibero.slurm")
    echo "  smoke test → job $SMOKE_JOB_ID"
    SMOKE_DEP="--dependency=afterok:${SMOKE_JOB_ID}"
fi

# ── OpenVLA jobs ──────────────────────────────────────────────────────────────
if ! $PI05_ONLY; then
    echo ""
    echo "--- OpenVLA Jobs ---"
    for SUITE in "${SUITES[@]}"; do
        # Long suite needs more time
        TIME="10:00:00"
        [[ "$SUITE" == "safelibero_long" ]] && TIME="14:00:00"
        SHORT="${SUITE##*_}"   # spatial, object, goal, long

        for LEVEL in "${LEVELS[@]}"; do
            JOB_NAME="openvla_${SHORT}_L${LEVEL}"
            JOB_ID=$(sbatch_cmd \
                --job-name="$JOB_NAME" \
                --time="$TIME" \
                --export="ALL,SUITE=$SUITE,LEVEL=$LEVEL" \
                "$SLURM_DIR/eval_openvla_safelibero.slurm")
            JOB_IDS["$JOB_NAME"]="$JOB_ID"
            echo "  $JOB_NAME → job $JOB_ID"
        done
    done
fi

# ── Pi0.5 jobs ────────────────────────────────────────────────────────────────
if ! $OPENVLA_ONLY; then
    echo ""
    echo "--- Pi0.5 Jobs${SMOKE_DEP:+ (depend on smoke: ${SMOKE_DEP##*:})} ---"
    for SUITE in "${SUITES[@]}"; do
        TIME="12:00:00"
        [[ "$SUITE" == "safelibero_long" ]] && TIME="16:00:00"
        SHORT="${SUITE##*_}"

        for LEVEL in "${LEVELS[@]}"; do
            JOB_NAME="pi05_${SHORT}_L${LEVEL}"
            JOB_ID=$(sbatch_cmd \
                ${SMOKE_DEP:+$SMOKE_DEP} \
                --job-name="$JOB_NAME" \
                --time="$TIME" \
                --export="ALL,SUITE=$SUITE,LEVEL=$LEVEL" \
                "$SLURM_DIR/eval_pi05_safelibero.slurm")
            JOB_IDS["$JOB_NAME"]="$JOB_ID"
            echo "  $JOB_NAME → job $JOB_ID"
        done
    done
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " All jobs submitted. Monitor with:"
echo "   squeue -u \$USER"
echo ""
echo " Logs at: /ocean/projects/cis250185p/asingal/slurm_logs/"
echo ""
echo " After all jobs finish, generate result tables:"
echo "   python scripts/aggregate_results.py"
echo "========================================================"
