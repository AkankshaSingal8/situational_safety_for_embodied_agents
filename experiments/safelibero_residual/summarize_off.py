"""Official SafeLIBERO episodes: TSR / CAR / SR per suite and level, pi0.5 vs residual arms.
    python summarize_off.py <tag>/<arm> ...            (pi05 is always included)
    EPS=all  python summarize_off.py final_pg/true     all 50 episodes per task and level (400 per suite)
    EPS=test (default)                                  test episodes 40-49 only (80 per suite)
Metrics as in the SafeLIBERO paper: TSR = task success, CAR = collision avoidance rate (no obstacle moved > 1 mm),
SR = safe success (success and no collision). Reads every runs_v2/<suite>/test/<arm>/eps*/ chunk.
Episodes 0-39 share their layouts (task-object positions) with the residual's training scenes; 40-49 do not."""
import glob, json, os, sys
import numpy as np

SUITES = os.environ.get("SUITES", "spatial,object,goal,long").split(",")
MODE = os.environ.get("EPS", "test")


def load(s, arm):
    rows = {}
    for f in glob.glob(f"runs_v2/{s}/test/{arm}/eps*/part_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if "error" not in r and (MODE == "all" or r["ep"] >= 40):
                rows[(r["level"], r["task"], r["ep"])] = r
    return rows


def st(rs):
    if not rs:
        return "-"
    return " / ".join(f"{v:.2f}" for v in (np.mean([r["success"] for r in rs]), 1 - np.mean([r["collide"] for r in rs]),
                                            np.mean([r["success"] and not r["collide"] for r in rs])))


arms = ["pi05"] + sys.argv[1:]
print(f"TSR / CAR / SR on official episodes ({'0-49' if MODE == 'all' else '40-49'}; keys common to all arms)")
for s in SUITES:
    D = [load(s, a) for a in arms]
    keys = sorted(set.intersection(*[set(d) for d in D]))
    counts = {a: len(d) for a, d in zip(arms, D)}
    if len(set(counts.values())) > 1 or len(keys) != max(counts.values()):
        print(f"WARNING {s}: arms ran different episode sets {counts}; table uses the {len(keys)} common ones")
    for lv in ("I", "II", None):
        ks = [k for k in keys if lv is None or k[0] == lv]
        print(f"{s:8s} {lv or 'all':3s} n={len(ks):3d}  " + "  |  ".join(f"{a}: {st([d[k] for k in ks])}" for a, d in zip(arms, D)))
