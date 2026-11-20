"""Seen vs novel table on the paired test scenes: TSR, CAR (any obstacle moved >1 mm), safe success.
    python summarize_pair.py pi05 [<residual tag>/<arm> ...]
Rows: per suite, and per seen->novel object type (pooled over suites). Scenes are paired by index."""
import collections, glob, json, sys
import numpy as np

import os
SUITES = os.environ.get("SUITES", "spatial,object,goal,long").split(",")


def load(s, tag, arm):
    rows = {}
    for f in glob.glob(f"runs_v2/{s}/{tag}/{arm}/all/part_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if "error" not in r:
                rows[r["scene"]] = r
    return rows


def stats(rs):
    if not rs:
        return "-"
    t = np.mean([r["success"] for r in rs]); c = np.mean([r["collide"] for r in rs])
    ss = np.mean([r["success"] and not r["collide"] for r in rs])
    return f"{t:.2f} / {c:.2f} / {ss:.2f}"


def load_scenes(path):  # same as eval_sl.load_scenes, without importing the simulator
    z = np.load(path, allow_pickle=True)
    return z["states"], json.loads(str(z["meta"]))


def main():
    arms = sys.argv[1:] or ["pi05"]
    for arm in arms:
        print(f"\n## {arm}   TSR / CAR / safe-success   (seen | novel, n)")
        by_type = collections.defaultdict(lambda: ([], []))
        for s in SUITES:
            seen, nov = load(s, "scenes_pair_seen", arm), load(s, "scenes_pair_novel", arm)
            meta = load_scenes(f"data_v2/{s}/scenes_pair_novel.npz")[1]
            ids = sorted(set(seen) & set(nov))
            if seen or nov:
                if not (len(seen) == len(nov) == len(meta)):
                    print(f"WARNING {s}: {arm} ran seen {len(seen)}, novel {len(nov)} of {len(meta)} paired scenes")
            print(f"{s:8s} {stats([seen[i] for i in ids]):18s} | {stats([nov[i] for i in ids]):18s} n={len(ids)}")
            for i in ids:
                k = f"{meta[i]['seen_obstacle'].replace('_small','').replace('_obstacle_1','')} -> " \
                    f"{meta[i]['obstacles'][0].replace('_obstacle_1','')}"
                by_type[k][0].append(seen[i]); by_type[k][1].append(nov[i])
        for k, (a, b) in sorted(by_type.items()):
            print(f"  {k:42s} {stats(a):18s} | {stats(b):18s} n={len(a)}")


if __name__ == "__main__":
    main()
