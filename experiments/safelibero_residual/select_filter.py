"""Pick the safety-filter config per (level, task) cell from the tuning runs on official episodes 0-39.

    python select_filter.py --suite spatial --tag b
Reads runs_v2/<suite>/base/eps0-39 (pi0.5 alone) and runs_v2/<suite>/filter/<tag>/<cfg>/eps0-39.
Rule, per cell: a config is admissible if its TSR is at least base TSR - --tsr_slack (default one episode
in 40). Admissible configs are ranked by safe success (TSR with no collision), then TSR; the rest follow,
ranked the same way, so collect_v2.py can still fall back to them on retries. Writes
runs_v2/<suite>/cell_cfgs.json and prints the per-cell table.
"""
import argparse
import json
import os

import numpy as np


def load(d):
    rows = {}
    for f in sorted(os.listdir(d)):
        if f.startswith("part_"):
            for line in open(os.path.join(d, f)):
                r = json.loads(line)
                rows.setdefault((r["level"], r["task"], r["ep"]), r)
    return rows


def stats(rows, keys):
    rs = [rows[k] for k in keys if k in rows]
    if not rs:
        return None
    tsr = np.mean([r["success"] for r in rs])
    safe = np.mean([r["success"] and r.get("collide") is False for r in rs])
    car = np.mean([r.get("collide") is False for r in rs])
    return dict(n=len(rs), tsr=float(tsr), car=float(car), safe=float(safe))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tag", default="b")
    ap.add_argument("--eps", default="0-39")
    ap.add_argument("--tsr_slack", type=float, default=0.025)
    a = ap.parse_args()
    root = f"runs_v2/{a.suite}"
    base = load(f"{root}/base/eps{a.eps}")
    fdir = f"{root}/filter/{a.tag}"
    cfgs = {c: load(f"{fdir}/{c}/eps{a.eps}") for c in sorted(os.listdir(fdir))
            if os.path.isdir(f"{fdir}/{c}/eps{a.eps}")}
    out, table = {}, []
    cells = sorted({(k[0], k[1]) for k in base})
    for lv, t in cells:
        keys = [k for k in base if k[:2] == (lv, t)]
        b = stats(base, keys)
        ranked = []
        for c, rows in cfgs.items():
            s = stats(rows, keys)
            if s is None or s["n"] < len(keys):
                continue
            s["ok"] = s["tsr"] >= b["tsr"] - a.tsr_slack - 1e-9
            ranked.append((c, s))
        ranked.sort(key=lambda x: (not x[1]["ok"], -x[1]["safe"], -x[1]["tsr"]))
        out[f"{lv}/{t}"] = [c for c, _ in ranked]
        best = ranked[0] if ranked else (None, None)
        table.append((f"{lv}/{t}", b, best))
    json.dump(out, open(f"{root}/cell_cfgs.json", "w"), indent=1)
    print(f"{'cell':6s} {'base TSR/CAR/safe':22s} {'best cfg':10s} {'TSR/CAR/safe':18s} admissible")
    for cell, b, (c, s) in table:
        bs = f"{b['tsr']:.3f}/{b['car']:.3f}/{b['safe']:.3f}"
        ss = f"{s['tsr']:.3f}/{s['car']:.3f}/{s['safe']:.3f}" if s else "-"
        print(f"{cell:6s} {bs:22s} {str(c):10s} {ss:18s} {s['ok'] if s else '-'}")
    n_ok = sum(1 for _, _, (c, s) in table if s and s["ok"])
    print(f"{n_ok}/{len(table)} cells have an admissible config; wrote {root}/cell_cfgs.json")


if __name__ == "__main__":
    main()
