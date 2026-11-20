"""Check residual training data before a run: labels match the rollout, and enough chunks carry a correction.

    python check_data.py data_v2/*/r0 data_v2/*/r1_* data_gso/* data_gso_r2/*

Per directory (every episode npz):
  label error   max |drive_tgt - (exec - base)| over the executed steps of every chunk (must be ~0)
  base-only     episodes without a driving sentence must have zero labels and exec == base
  meta          safe_success must be true for every kept driven episode (collection keeps only those)
  corrections   corrected episodes / chunks, mean |correction| per corrected chunk, GSO objects with a correction
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def one(f):
    with np.load(f, allow_pickle=True) as z:
        m = json.loads(str(z["meta"]))
        E, B, T, L = z["exec"], z["base"], z["drive_tgt"], z["exec_len"]
    err, dev = 0.0, 0.0
    for c in range(len(L)):
        k = int(L[c])
        d = E[c, :k, :3] - B[c, :k, :3]
        err = max(err, float(np.abs(T[c, :k] - d).max()) if m.get("drive") else float(np.abs(T[c]).max()))
        dev = max(dev, float(np.abs(d).max()))
    nzc = np.abs(T).max(axis=(1, 2)) > 1e-6
    mag = np.abs(T[nzc]).sum(-1).max(-1) if nzc.any() else np.zeros(0)
    return dict(driven=bool(m.get("drive")), safe=bool(m.get("safe_success", True)), err=err,
                base_dev=0.0 if m.get("drive") else dev, chunks=len(L), nz=int(nzc.sum()), mag=mag.tolist(),
                gso=m.get("gso"))


def main():
    dirs = [d for p in sys.argv[1:] for d in sorted(glob.glob(p)) if os.path.isdir(d)]
    tot = dict(ep=0, corr=0, chunks=0, nz=0)
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 16))) as ex:
        for d in dirs:
            files = sorted(glob.glob(f"{d}/ep_*.npz"))
            if not files:
                continue
            R = list(ex.map(one, files, chunksize=8))
            drv = [r for r in R if r["driven"]]
            bad = sum(r["err"] > 1e-4 for r in R) + sum(r["base_dev"] > 1e-5 for r in R)
            unsafe = sum(not r["safe"] for r in drv)
            corr = [r for r in drv if r["nz"]]
            ch, nz = sum(r["chunks"] for r in R), sum(r["nz"] for r in R)
            mags = [x for r in corr for x in r["mag"]]
            objs = {r["gso"] for r in R if r["gso"]}
            objs_c = {r["gso"] for r in corr if r["gso"]}
            print(f"{d:28s} ep={len(R):5d} driven={len(drv):5d} corrected={len(corr):5d} "
                  f"chunks={ch:6d} corrected_chunks={nz:5d} ({nz / max(ch, 1):.3f}) "
                  f"|corr|={np.mean(mags) if mags else 0:.3f} max_label_err={max(r['err'] for r in R):.1e} "
                  f"bad={bad} unsafe_kept={unsafe}"
                  + (f" objects={len(objs)} with_correction={len(objs_c)}" if objs else ""), flush=True)
            tot["ep"] += len(R); tot["corr"] += len(corr); tot["chunks"] += ch; tot["nz"] += nz
    print(f"TOTAL episodes {tot['ep']}, corrected {tot['corr']}, chunks {tot['chunks']}, corrected chunks {tot['nz']} "
          f"({tot['nz'] / max(tot['chunks'], 1):.3f})")


if __name__ == "__main__":
    main()
