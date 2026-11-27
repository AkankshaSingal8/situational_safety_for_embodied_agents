"""Paired failure breakdown, pi0.5 vs ex2, per suite/table/level.
    python fail_check.py [--ex ex2_1gpu_bs64]
For each paired run: categories of ex2 non-SR runs (collided vs avoided-but-task-failed), and transitions
pi0.5 SR -> ex2 not SR ("broke") and pi0.5 not SR -> ex2 SR ("fixed")."""
import argparse
from collections import Counter

from make_tables import SUITES, load

ap = argparse.ArgumentParser()
ap.add_argument("--ex", default="ex2_1gpu_bs64")
ap.add_argument("--arm", default="true")
a = ap.parse_args()
key = lambda r: (r["level"], r["task"], r.get("scene", r.get("ep")), r.get("ep"))
sr = lambda r: r["success"] and not r["collide"]
print("table suite lvl n | ex2 fail: collide / avoided-taskfail | pi0.5 fail: collide / taskfail | fixed broke"
      " | broke: ex2 collide / ex2 taskfail")
for sub in ["test", "scenes_newobj25", "scenes_newlayout"]:
    for s in SUITES:
        P = {key(r): r for r in load(f"runs_v2/{s}/{sub}/pi05/*/part_*.jsonl")}
        E = {key(r): r for r in load(f"runs_v2/{s}/{sub}/{a.ex}/{a.arm}/*/part_*.jsonl")}
        for L in ("I", "II"):
            ks = [k for k in E if k in P and k[0] == L]
            c = Counter()
            for k in ks:
                p, e = P[k], E[k]
                if not sr(e):
                    c["ec" if e["collide"] else "ef"] += 1
                if not sr(p):
                    c["pc" if p["collide"] else "pf"] += 1
                if sr(p) and not sr(e):
                    c["broke"] += 1
                    c["bc" if e["collide"] else "bf"] += 1
                if sr(e) and not sr(p):
                    c["fixed"] += 1
            print(f"{sub[:14]:14s} {s:7s} {L:2s} {len(ks):3d} | {c['ec']:3d} / {c['ef']:3d} | {c['pc']:3d} / {c['pf']:3d}"
                  f" | {c['fixed']:3d} {c['broke']:3d} | {c['bc']:3d} / {c['bf']:3d}")
