"""Paired comparison of eval arms on the episodes they share.

    python compare.py runs/base/eval_official runs/gate/oracle_c0 [--eps 0-24] [--levels I,II]
Prints TSR / CAR / safe-success for each arm and exact McNemar tests against the first arm
(b = arm better, c = first arm better). For generated two-obstacle scenes it also prints
the named-obstacle and other-obstacle hit rates.
"""
import argparse
import json
import os
from math import comb

import numpy as np


def load(d):
    """One row per episode. If an episode was run twice (two jobs writing the same dir concurrently, as
    happened for res_i1b_held / res_i1b_wrong on 2026-11-01), keep the first row in file order (part_0..part_4, line order; rows carry no timestamps),
    a rule fixed before looking at those results, and report the count."""
    rows, dup = {}, 0
    for f in sorted(os.listdir(d)):
        if f.startswith("part_"):
            for line in open(os.path.join(d, f)):
                r = json.loads(line)
                k = ("scn", r["scene"]) if "scene" in r else (r["level"], r["task"], r["ep"])
                if k in rows:
                    dup += 1
                    continue
                rows[k] = r
    if dup:
        print(f"[note] {d}: {dup} duplicate episode rows ignored (first in file order kept)")
    return rows


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


METRICS = {"TSR": lambda r: r["success"], "CAR": lambda r: r.get("collide") is False,
           "safe&succ": lambda r: r["success"] and r.get("collide") is False,
           "noContact": lambda r: r.get("robot_contact") is False,
           "succ&noContact": lambda r: r["success"] and r.get("robot_contact") is False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--eps", default="")
    ap.add_argument("--levels", default="I,II")
    ap.add_argument("--json", default="")
    ap.add_argument("--obstacles", default="", help="keep episodes whose (first) obstacle is in this comma list; "
                                                   "prefix with ! to exclude instead")
    a = ap.parse_args()
    arms = [load(d) for d in a.dirs]
    keys = set(arms[0])
    for x in arms[1:]:
        keys &= set(x)
    if a.eps:
        lo, hi = map(int, a.eps.split("-"))
        keys = {k for k in keys if k[0] == "scn" or lo <= k[2] <= hi}
    keys = {k for k in keys if k[0] == "scn" or k[0] in a.levels.split(",")}
    if a.obstacles:
        neg = a.obstacles.startswith("!")
        sel = set(a.obstacles.lstrip("!").split(","))
        first = lambda r: (r.get("obstacles") or ["?"])[0]  # noqa: E731
        keys = {k for k in keys if (first(arms[0][k]) in sel) != neg}
    keys = sorted(keys, key=str)
    out = {"n": len(keys), "arms": {}}
    print(f"{len(keys)} shared episodes")
    for d, x in zip(a.dirs, arms):
        rs = [x[k] for k in keys]
        res = {m: float(np.mean([f(r) for r in rs])) for m, f in METRICS.items()}
        res["errors"] = sum("error" in r for r in rs)
        if any(len(r.get("obstacles", [])) > 1 for r in rs):
            res["named_hit"] = float(np.mean([r.get("named_hit", False) for r in rs]))
            res["other_hit"] = float(np.mean([r.get("other_hit", False) for r in rs]))
            res["named_contact"] = float(np.mean([r.get("named_contact", False) for r in rs]))
            res["other_contact"] = float(np.mean([r.get("other_contact", False) for r in rs]))
        if d != a.dirs[0]:
            for m, f in METRICS.items():
                b = sum(f(x[k]) and not f(arms[0][k]) for k in keys)
                c = sum(f(arms[0][k]) and not f(x[k]) for k in keys)
                res[f"{m}_b/c"] = f"{b}/{c}"; res[f"{m}_p"] = mcnemar(b, c)
        out["arms"][d] = res
        line = "  ".join(f"{m} {res[m]:.3f}" for m in METRICS)
        extra = "  ".join(f"{m} {res[m + '_b/c']} p={res[m + '_p']:.2g}" for m in METRICS if m + "_p" in res)
        nh = (f"  named_hit {res['named_hit']:.3f} other_hit {res['other_hit']:.3f} named_contact "
              f"{res['named_contact']:.3f} other_contact {res['other_contact']:.3f}") if "named_hit" in res else ""
        print(f"{d:45s} {line}{nh}  errors {res['errors']}\n{'':45s} {extra}" if extra else f"{d:45s} {line}{nh}  errors {res['errors']}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
