#!/bin/bash
#SBATCH --job-name=ls_unzip
#SBATCH --partition=RM-shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# LIBERO-Safety assets extraction (15.5GB, ~900k files). Login-node attempts
# kept getting killed; run as a CPU job. Chains the smoke test on success.
set -euo pipefail
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
ZIP=/ocean/projects/cis250185p/asingal/.hf_cache/hub/datasets--LIBERO-Safety--libero_safety_assets/snapshots/94ccca773d1bab760d1aecec4f62c8d4b8797d40/assets.zip
cd "$LS"
unzip -oq "$ZIP" -d assets_unzip_tmp
rsync -a assets_unzip_tmp/assets/ libero/libero/assets/
rm -rf assets_unzip_tmp
touch libero/libero/assets/UNZIP_DONE
sbatch "$WT/slurm/libero_safety_smoke.slurm"
echo "UNZIP_CHAIN_OK"
