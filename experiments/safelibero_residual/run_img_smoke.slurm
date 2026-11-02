#!/bin/bash
#SBATCH -J slres_ismoke
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 00:45:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Smoke test of the image residual path end to end on tiny data.
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safelibero_residual
PY=/ocean/projects/cis250185p/asingal/envs/openvla_libero_merged/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=$PWD
CK=runs/base/ema_last.pt; SM=runs/smoke_img; rm -rf $SM data/smoke_img_*; mkdir -p $SM
$PY checks.py 2>&1 | grep -E "^ok|PASSED|Error|assert" || true
$PY collect_residual.py --ckpt $CK --scenes data/scenes_train.npz --out data/smoke_img_or --workers 2 --limit 8 --save_images
$PY collect_residual.py --ckpt $CK --scenes data/scenes_train.npz --out data/smoke_img_dag --workers 2 --limit 8 --save_images \
    --residual runs/res_text_c0/residual.pt --seed 1
$PY - <<'P'
import numpy as np, glob
for d in ["data/smoke_img_or", "data/smoke_img_dag"]:
    for f in sorted(glob.glob(d + "/part_*.npz")):
        z = np.load(f); a = np.load(f.replace("part_", "agent_").replace(".npz", ".npy")); w = np.load(f.replace("part_", "wrist_").replace(".npz", ".npy"))
        print(f, z["feats"].shape, z["chunk_scene"][:3], a.shape, a.dtype, w.shape, a.mean().round(1))
    print(open(sorted(glob.glob(d + "/episodes_*.json"))[0]).read()[:300])
P
# image orientation: env frame used by the base vs the stored frame
$PY - <<'P'
import numpy as np, glob
packed = "/ocean/projects/cis250185p/asingal/data_libero/packed_spatial"
d = np.load(packed + "/agent.npy", mmap_mode="r")[:2000:50].astype(np.float32).mean(0)
a = np.load("data/smoke_img_or/agent_0.npy")[:20].astype(np.float32).mean(0)
c = lambda x, y: np.corrcoef(x.ravel(), y.ravel())[0, 1]
print("orientation corr stored-vs-demo", round(c(a, d), 3), "flipped", round(c(a[::-1, ::-1], d), 3))
P
for V in text_img notext_img; do
  $PY train_residual.py --data data/smoke_img_or,data/smoke_img_dag --out $SM/res_$V --variant $V --steps 300 --bs 64 \
      --scenes data/scenes_train.npz --holdout_layouts 1-1 2>&1 | grep -v -i warn
done
$PY train_residual.py --data data/smoke_img_or,data/smoke_img_dag --out $SM/res_text --variant text --steps 300 \
    --scenes data/scenes_train.npz --holdout_layouts 1-1 2>&1 | grep -v -i warn
$PY eval_sl.py --ckpt $CK --workers 2 --out $SM/eval --arm residual --residual $SM/res_text_img/residual.pt --text_mode true --eps 20-20 --tasks 0,1 --levels I
cat $SM/eval/part_*.jsonl | cut -c1-300
echo ALL_DONE
