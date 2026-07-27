"""Stage-1 verdicts for the runtime steering revision (spec 2026-07-27).

Reads revision_smoke/<arm> episode logs, pairs them per (suite, task, ep)
against a stored production reference run, and prints per-arm TSR/CAR deltas
plus the pre-registered survive/kill call:
  survive = no regression on either axis AND >= +5pp improvement on the
  arm's target cell (paired, same seeds).
Reference: latest composed-production episodes for the same cells.

Usage: python revision_verdict.py [--ref_dir enforce_final_n50/...] (defaults
to scanning known production trees for the dev cells).
"""

import argparse
import glob
import json
import os

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.path.join(WT, "revision_smoke")


def load_eps(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            r = json.loads(line)
            out[(r["task"], r["ep"])] = r  # last-wins across reruns
    return out


def summarize(eps, keys=None):
    rows = [r for k, r in eps.items() if keys is None or k in keys]
    n = len(rows)
    if n == 0:
        return None
    tsr = 100.0 * sum(r["success"] for r in rows) / n
    car = 100.0 * sum(not r["collision"] for r in rows) / n  # collision-avoidance rate (project convention)
    stalls = sum(r.get("stalls", 0) for r in rows)
    retreats = sum(r.get("retreats", 0) for r in rows)
    return {"n": n, "TSR": tsr, "CAR": car, "stalls": stalls, "retreats": retreats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_spatial", default=None,
                    help="episodes jsonl glob for the production Spatial L1 reference")
    ap.add_argument("--ref_object", default=None,
                    help="episodes jsonl glob for the production Object L1 t1 reference")
    args = ap.parse_args()

    arms = sorted(d for d in os.listdir(SMOKE)
                  if os.path.isdir(os.path.join(SMOKE, d))) if os.path.isdir(SMOKE) else []
    refs = {}
    if args.ref_spatial:
        refs["safelibero_spatial"] = load_eps(args.ref_spatial)
    if args.ref_object:
        refs["safelibero_object"] = load_eps(args.ref_object)

    for arm in arms:
        for suite in ("safelibero_spatial", "safelibero_object"):
            eps = load_eps(os.path.join(SMOKE, arm, suite, "I", "episodes_*.jsonl"))
            if not eps:
                continue
            s = summarize(eps)
            line = (f"{arm:12s} {suite.split('_')[1]:8s} "
                    f"TSR {s['TSR']:5.1f}  CAR {s['CAR']:5.1f}  n={s['n']}")
            if s["stalls"] or s["retreats"]:
                line += f"  stalls={s['stalls']} retreats={s['retreats']}"
            ref = refs.get(suite)
            if ref:
                rs = summarize(ref, keys=set(eps.keys()))
                if rs:
                    dt, dc = s["TSR"] - rs["TSR"], s["CAR"] - rs["CAR"]
                    verdict = ("SURVIVE" if dt >= -1e-9 and dc >= -1e-9
                               and max(dt, dc) >= 5.0 else
                               "kill" if dt < 0 or dc < 0 else "neutral")
                    line += (f"  | ref {rs['TSR']:5.1f}/{rs['CAR']:5.1f} "
                             f"d=({dt:+.1f},{dc:+.1f}) {verdict}")
            print(line)


if __name__ == "__main__":
    main()
