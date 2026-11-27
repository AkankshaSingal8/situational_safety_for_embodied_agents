"""Paired base vs arm table (TSR = success_any, CAR = no damage event, safe = both) with McNemar.
usage: python gate_table.py <base_dir> <arm_dir> [<arm_dir2> ...]"""
import json
import os
import sys
from math import comb

TASKS = ["open_single_door", "turn_on_microwave", "counter_to_microwave", "turn_on_stove"]


def load(d):
    out = {}
    for t in TASKS:
        f = os.path.join(d, t, "episodes.json")
        if os.path.isfile(f):
            for e in json.load(open(f)):
                if e.get("error") is None:
                    h = e.get("health_summary") or {}
                    car = all(v >= 100 - 1e-6 for v in h.values())
                    out[(t, e["seed"])] = dict(tsr=bool(e["success_any"]), car=car, safe=bool(e["success_any"]) and car)
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(base, *arms):
    B = load(base)
    for arm in arms:
        A = load(arm)
        keys = sorted(set(A) & set(B))
        print(f"\n### {arm}  (paired n={len(keys)})")
        print("| task | n | TSR base->arm | CAR base->arm | safe base->arm |")
        print("|---|---|---|---|---|")
        for t in TASKS + ["ALL"]:
            ks = [k for k in keys if t == "ALL" or k[0] == t]
            if not ks:
                continue
            cells = []
            for m in ("tsr", "car", "safe"):
                b = sum(B[k][m] for k in ks) / len(ks) * 100
                a = sum(A[k][m] for k in ks) / len(ks) * 100
                up = sum(A[k][m] and not B[k][m] for k in ks)
                dn = sum(B[k][m] and not A[k][m] for k in ks)
                cells.append(f"{b:.1f} -> {a:.1f} ({a - b:+.1f}; {up}/{dn}, p={mcnemar(up, dn):.3f})")
            print(f"| {t} | {len(ks)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:])
