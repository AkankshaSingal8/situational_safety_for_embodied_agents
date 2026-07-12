#!/bin/bash
# Submit the CAVF smoke test and leave an audit trail even if Slurm rejects it.
set -euo pipefail

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE="$ROOT/.worktrees/cavf-smoke-poc-20260712"
RUN_ROOT="$WORKTREE/cavf_static_smoke"
LOG_DIR="$RUN_ROOT/logs"
SCRIPT="$WORKTREE/vlm_pipeline/slurm_cavf_openvla_smoke.sh"
SUBMISSION_LOG="$LOG_DIR/submission.log"

mkdir -p "$LOG_DIR"
{
  printf 'submitted_at=%s\n' "$(date -Is)"
  printf 'worktree=%s\n' "$WORKTREE"
  printf 'branch=%s\n' "$(git -C "$WORKTREE" branch --show-current)"
  printf 'commit=%s\n' "$(git -C "$WORKTREE" rev-parse --short HEAD)"
  printf 'script=%s\n' "$SCRIPT"
} | tee "$SUBMISSION_LOG"

if ! raw_job_id=$(sbatch --parsable \
  --output="$LOG_DIR/slurm_%j.out" \
  --error="$LOG_DIR/slurm_%j.err" \
  "$SCRIPT" 2>&1); then
  printf 'submission_failed=%s\n' "$raw_job_id" | tee -a "$SUBMISSION_LOG"
  exit 1
fi

printf 'slurm_response=%s\n' "$raw_job_id" | tee -a "$SUBMISSION_LOG"
job_id=\${raw_job_id%%;*}
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  printf 'submission_failed=invalid_job_id\n' | tee -a "$SUBMISSION_LOG"
  exit 1
fi

printf 'job_id=%s\n' "$job_id" | tee -a "$SUBMISSION_LOG"
squeue -j "$job_id" -o '%.18i %.2t %.10M %.30j' | tee -a "$SUBMISSION_LOG" || true
