#!/bin/bash
# Unprivileged apptainer (conda-forge) + B1K container as a writable sandbox on /data.
# pleiades has no apptainer/singularity, but unprivileged user namespaces are enabled on compute nodes.
set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/asingal2/micromamba
MM=$HOME/.local/bin/micromamba
A=/data/asingal2/envs/apptainer
[ -x $A/bin/apptainer ] || $MM create -y -p $A -c conda-forge apptainer squashfuse fuse-overlayfs
export PATH=$A/bin:$PATH
apptainer --version
export APPTAINER_CACHEDIR=/data/asingal2/apptainer_cache APPTAINER_TMPDIR=/data/asingal2/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
SB=/data/asingal2/oopsieverse/b1k_sandbox
if [ ! -d $SB/opt/miniconda3 ]; then
  rm -rf $SB
  apptainer build --fakeroot --sandbox $SB /data/asingal2/oopsieverse/pleiades_jobs/b1k_container.def
fi
apptainer exec --nv $SB bash -c 'ldd --version | head -1; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; /opt/miniconda3/bin/conda --version'
echo SANDBOX_DONE
