#!/bin/bash
# Set up and run the ex2 SafeLIBERO evals (Tables 1a, 1b, 2, 3; pi0.5 and ex2 arms) on a plain GPU machine.
# No SLURM and no Claude needed. Run it on the GPU node itself:
#
#   export GH_TOKEN=...        # only if the GitHub repo is private (read access is enough)
#   bash replicate_ex2_eval.sh # all steps; or STEPS="eval tables" bash replicate_ex2_eval.sh
#
# Steps (STEPS, default "clone envs assets eval tables"):
#   clone   clone the repo branch + the SafeLIBERO and vlsa-aegis submodules into $WORK/repo,
#           rewrite the cluster paths in the scripts to $WORK
#   envs    build two Python envs under $WORK/data/envs: slres (sim, py3.10) and openpi (pi0.5 server, py3.11)
#   assets  pi0.5 checkpoints (pi05_libero, pi05_base), the ex2 checkpoint and the 4 base-policy files from the
#           GitHub release, the GSO meshes (mujoco_scanned_objects), and the Table 2/3 scene files
#   eval    per suite: start the pi0.5 server and the ex2 server, run pi0.5 alone and pi0.5 + ex2 on
#           Table 1 (official eps 0-49), Table 2 (25 new objects) and Table 3 (held-out layouts).
#           Suites run in parallel, one per GPU (round-robin if fewer than 4 GPUs).
#   tables  print the 4 tables (also written to $WORK/ex2_tables.md)
#
# Needs: Linux x86_64, NVIDIA driver for CUDA 12, git, git-lfs, curl, bzip2, tar. ~60 GB disk.
# GPU: one GPU with >= 40 GB per suite runs both servers (about 16-20 GB in use, 6 sim workers);
# 24 GB GPUs ran out of memory on our cluster. Each suite needs ~8 CPU cores.
# Time: about 3-5 h per suite on one RTX 6000 Ada (2 arms x ~1,000 episodes).
set -euo pipefail

WORK=${WORK:-$HOME/ex2_eval}
REPO_URL=${REPO_URL:-https://github.com/AkankshaSingal8/situational_safety_for_embodied_agents.git}
BRANCH=${BRANCH:-safelibero-residual-ex2}
RELEASE=${RELEASE:-ex2-v1}
STEPS=${STEPS:-clone envs assets eval tables}
SUITES=${SUITES:-spatial object goal long}
WORKERS=${WORKERS:-6}

REPO=$WORK/repo
DATA=$WORK/data
EXP=$REPO/experiments/safelibero_residual
OWNER_REPO=$(echo "$REPO_URL" | sed -E 's#.*github.com[/:]##; s#\.git$##')
OLD_REPO=/home/asingal2/safety/situational_safety_for_embodied_agents
OLD_DATA=/data/asingal2
mkdir -p "$WORK" "$DATA/slurm_logs" "$DATA/envs" "$HOME/.local/bin"
export PATH=$HOME/.local/bin:$PATH
log() { echo "[$(date +%H:%M:%S)] $*"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
auth_url() { if [ -n "${GH_TOKEN:-}" ]; then echo "$REPO_URL" | sed "s#https://#https://x-access-token:${GH_TOKEN}@#"; else echo "$REPO_URL"; fi; }

# ------------------------------------------------------------------ clone
if has clone; then
  log "clone $BRANCH -> $REPO"
  command -v git-lfs >/dev/null || { echo "git-lfs is required (SafeLIBERO assets are in LFS)"; exit 1; }
  git lfs install --skip-repo
  if [ ! -d "$REPO/.git" ]; then git clone --branch "$BRANCH" --single-branch "$(auth_url)" "$REPO"; fi
  cd "$REPO"
  git remote set-url origin "$REPO_URL"   # keep the token out of .git/config
  git submodule update --init SafeLIBERO vlsa-aegis
  (cd SafeLIBERO && git lfs pull)
  log "rewrite cluster paths -> $WORK"
  # scripts hard-code the cluster's repo and data paths; point them at this machine
  grep -rlE "$OLD_REPO|$OLD_DATA" "$EXP" setup/safelibero --include='*.py' --include='*.sh' --include='*.slurm' \
    | xargs -r sed -i -e "s#$OLD_REPO#$REPO#g" -e "s#$OLD_DATA#$DATA#g"
  # SLURM-only lines in the eval script are harmless; it only needs SLURM_JOB_ID for log names
fi

# ------------------------------------------------------------------ envs
if has envs; then
  if ! command -v micromamba >/dev/null; then
    log "install micromamba"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME/.local" bin/micromamba
  fi
  if ! command -v uv >/dev/null; then
    log "install uv"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
  fi
  log "build envs (slres + openpi); takes 20-40 min"
  # same script that built the cluster envs (pins: torch 2.2.2 cu118, mujoco 3.2.3, robosuite 1.4.1, numpy 1.24.4;
  # openpi via uv sync). Frozen package lists from the cluster are in env/*_freeze.txt if something drifts.
  REPO_ROOT=$REPO ENVS=$DATA/envs MAMBA_ROOT_PREFIX=$DATA/micromamba bash "$REPO/setup/safelibero/pleiades_slres_env.sh"
fi

PY=$DATA/envs/slres/bin/python
PYO=$DATA/envs/openpi/bin/python

# ------------------------------------------------------------------ assets
if has assets; then
  cd "$EXP"
  log "pi0.5 checkpoints (pi05_libero, pi05_base) -> $DATA/openpi_cache"
  bash run_pi05_server.sh download

  log "release $RELEASE assets: ex2 best.pkl + base-policy files"
  mkdir -p runs_v2/expert/ex2_1gpu_bs64
  "$PY" - "$OWNER_REPO" "$RELEASE" "$EXP" <<'EOF'
import json, os, sys, urllib.request
owner_repo, tag, exp = sys.argv[1:]
tok = os.environ.get("GH_TOKEN")
def req(url, accept="application/vnd.github+json"):
    r = urllib.request.Request(url, headers={"Accept": accept, **({"Authorization": f"Bearer {tok}"} if tok else {})})
    return urllib.request.urlopen(r)
rel = json.load(req(f"https://api.github.com/repos/{owner_repo}/releases/tags/{tag}"))
dest = {"best.pkl": "runs_v2/expert/ex2_1gpu_bs64/best.pkl"}
for s in ["spatial", "object", "goal", "long"]:
    dest[f"{s}_ema_last.pt"] = f"runs_scale/{s}/base/ema_last.pt"
for a in rel["assets"]:
    if a["name"] not in dest:
        continue
    out = os.path.join(exp, dest[a["name"]])
    if os.path.exists(out) and os.path.getsize(out) == a["size"]:
        print("have", out); continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print("download", a["name"], f"{a['size']/1e6:.0f} MB", flush=True)
    with req(a["url"], "application/octet-stream") as r, open(out + ".part", "wb") as f:
        while (b := r.read(1 << 24)):
            f.write(b)
    os.replace(out + ".part", out)
missing = [k for k, v in dest.items() if not os.path.exists(os.path.join(exp, v))]
sys.exit(f"missing release assets: {missing}" if missing else 0)
EOF

  log "GSO meshes (kevinzakka/mujoco_scanned_objects, CC-BY 4.0) -> $DATA/assets/gso_models.tar"
  mkdir -p "$DATA/assets"
  if [ ! -f "$DATA/assets/gso_models.tar" ]; then
    [ -d "$DATA/assets/mujoco_scanned_objects" ] || \
      git clone --depth 1 https://github.com/kevinzakka/mujoco_scanned_objects "$DATA/assets/mujoco_scanned_objects"
    (cd "$DATA/assets/mujoco_scanned_objects" && git lfs pull 2>/dev/null || true; tar cf "$DATA/assets/gso_models.tar" models)
  fi

  log "Table 2/3 scene files"
  for S in spatial object goal long; do
    mkdir -p data_gso/$S data_v2/$S
    cp eval_scenes/$S/scenes_newobj25.npz data_gso/$S/
    cp eval_scenes/$S/scenes_newlayout.npz data_v2/$S/
  done
  log "check: 25 test objects present in the mesh tar"
  "$PY" - <<'EOF'
import json, tarfile
objs = json.load(open("runs_v2/gso_test25.json"))
names = {n.split("/")[1] for n in tarfile.open(__import__("gso_objects").TAR).getnames() if n.count("/") >= 1}
miss = [o for o in objs if o not in names]
print("missing:", miss) if miss else print("all 25 test objects found")
EOF
fi

# ------------------------------------------------------------------ eval
if has eval; then
  cd "$EXP"
  EX=runs_v2/expert/ex2_1gpu_bs64/best.pkl
  [ -f "$EX" ] || { echo "missing $EX (run the assets step)"; exit 1; }
  NGPU=$(nvidia-smi -L | wc -l); [ "$NGPU" -ge 1 ] || { echo "no GPU visible"; exit 1; }
  log "eval on $NGPU GPU(s); logs in $DATA/slurm_logs"
  run_suite() {
    local S=$1 G=$2
    for SC in "" "../../data_gso/$S/scenes_newobj25" "scenes_newlayout"; do
      local tag=${SC:-official}; tag=$(basename "$tag")
      log "suite $S gpu $G: $tag (pi0.5 + ex2)"
      # ARMS="base true": pi0.5 alone, then pi0.5 + ex2 with the correct safety sentence, same servers and settings
      CUDA_VISIBLE_DEVICES=$G SLURM_JOB_ID=local_${S}_$tag S=$S ARMS="base true" EX=$EX EPS=0-49 SCENES=$SC \
        WORKERS=$WORKERS bash run_eval_pg.slurm > "$DATA/slurm_logs/eval_${S}_${tag}.out" 2>&1 \
        || { log "suite $S $tag FAILED; see $DATA/slurm_logs/eval_${S}_${tag}.out"; return 1; }
    done
    log "suite $S done"
  }
  i=0; pids=()
  for S in $SUITES; do
    G=$((i % NGPU))
    if [ $i -ge "$NGPU" ]; then wait "${pids[$((i - NGPU))]}" || true; fi   # one suite per GPU at a time
    run_suite "$S" "$G" & pids+=($!); i=$((i + 1))
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -eq 0 ] || log "some suites failed; tables will show 'pending' for them"
fi

# ------------------------------------------------------------------ tables
if has tables; then
  cd "$EXP"
  "$PY" make_tables.py --ex ex2_1gpu_bs64 | tee "$WORK/ex2_tables.md"
  log "tables written to $WORK/ex2_tables.md"
fi
