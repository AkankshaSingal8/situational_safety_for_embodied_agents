"""Held-out GSO objects on test layouts 40-49 (data_gso/<suite>/scenes_test.npz): TSR / CAR / SR per suite,
per level, and per object category (pooled), pi0.5 vs residual arms.
    python summarize_gso.py <tag>/<arm> ...        (pi05 is always included)
CAR = collision avoidance rate (no obstacle moved > 1 mm), SR = success and no collision."""
import collections
import glob
import json
import os
import sys

import numpy as np

SUITES = os.environ.get("SUITES", "spatial,object,goal,long").split(",")
# SCN=scenes_test (default, data_gso/<suite>/scenes_test.npz); SCN=scenes_newobj or SCN=scenes_newlayout read
# data_gso/<suite>/<SCN>.npz or data_v2/<suite>/<SCN>.npz, whichever exists
SCN = os.environ.get("SCN", "scenes_test")


def load(s, arm):
    rows = {}
    for f in glob.glob(f"runs_v2/{s}/{SCN}/{arm}/all/part_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if "error" not in r:
                rows[r["scene"]] = r
    return rows


def st(rs):
    if not rs:
        return "-"
    return " / ".join(f"{v:.2f}" for v in (np.mean([r["success"] for r in rs]), 1 - np.mean([r["collide"] for r in rs]),
                                            np.mean([r["success"] and not r["collide"] for r in rs])))


arms = ["pi05"] + sys.argv[1:]
print("TSR / CAR / SR, held-out GSO objects on test layouts 40-49 (scenes common to all arms)")
cat = collections.defaultdict(lambda: [[] for _ in arms])
for s in SUITES:
    p = f"data_gso/{s}/{SCN}.npz"
    if not os.path.exists(p):
        p = f"data_v2/{s}/{SCN}.npz"
    if not os.path.exists(p):
        continue
    meta = json.loads(str(np.load(p, allow_pickle=True)["meta"]))
    D = [load(s, a) for a in arms]
    keys = sorted(set.intersection(*[set(d) for d in D])) if D else []
    counts = {a: len(d) for a, d in zip(arms, D)}
    if len(set(counts.values())) > 1 or len(keys) != len(meta):
        print(f"WARNING {s}: arms ran {counts} of {len(meta)} scenes; table uses the {len(keys)} common ones")
    for lv in ("I", "II", None):
        ks = [k for k in keys if lv is None or meta[k]["level"] == lv]
        print(f"{s:8s} {lv or 'all':3s} n={len(ks):3d}  " + "  |  ".join(f"{a}: {st([d[k] for k in ks])}" for a, d in zip(arms, D)))
    for k in keys:
        for j, d in enumerate(D):
            cat[meta[k].get("category", "libero obstacle")][j].append(d[k])
print("\nper category (pooled over suites)")
for c, L in sorted(cat.items(), key=lambda x: -len(x[1][0])):
    print(f"  {c:18s} n={len(L[0]):3d}  " + "  |  ".join(f"{a}: {st(l)}" for a, l in zip(arms, L)))
