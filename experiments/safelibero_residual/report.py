"""Suite- and task-level TSR / CAR / safe-success tables for the scale-up (SCALEUP.md).

    python report.py --arm "flow base=base/eval_official" --arm "pi0.5 libero=pi05_libero/eval_official" \
                     --eps 0-49 --out report_bases.md
    python report.py --arm "base=flow/test/base" --arm "residual=flow/test/s{seed}/res_true" \
                     --arm "no-text=flow/test/s{seed}/notext" --eps 25-49 --out report_test_flow.md

Each arm is LABEL=PATH, where PATH is relative to runs_scale/<suite>/. A PATH containing {seed} is expanded to
seeds 0,1,2. Every seed is reported: the tables show mean ± sd over seeds, and per-seed values are listed below them.
CIs are 95% Wilson intervals on single runs. McNemar p-values are paired against the first arm on the same
episodes: exact two-sided, per seed, and Holm-corrected over the 4 suites. A missing suite or seed is
reported as missing, never filled in.
"""
import argparse
import os
from math import sqrt

import numpy as np

from compare import load, mcnemar

SUITES = ["spatial", "object", "goal", "long"]
SEEDS = [0, 1, 2]
MET = {"TSR": lambda r: bool(r["success"]),
       "CAR": lambda r: r.get("collide") is False,
       "safe": lambda r: bool(r["success"]) and r.get("collide") is False}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def eps_set(s):
    lo, hi = map(int, s.split("-"))
    return set(range(lo, hi + 1))


def rows_for(root, suite, path, eps):
    d = os.path.join(root, suite, path)
    if not os.path.isdir(d):
        return None
    r = load(d)
    return {k: v for k, v in r.items() if k[0] in ("I", "II") and k[2] in eps}


def rate(rows, m, sel=lambda k: True):
    ks = [k for k in rows if sel(k)]
    return sum(MET[m](rows[k]) for k in ks), len(ks)


def holm(ps):
    order = np.argsort(ps)
    out, run = [None] * len(ps), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        out[j] = run
    return out


def fmt_cell(vals, ns):
    """vals: per-seed rates. One seed: rate [CI]. Several: mean ± sd (n per seed)."""
    if not vals:
        return "—"
    if len(vals) == 1:
        lo, hi = wilson(round(vals[0] * ns[0]), ns[0])
        return f"{vals[0]:.3f} [{lo:.2f},{hi:.2f}]"
    return f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_scale")
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--eps", default="25-49")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    eps = eps_set(a.eps)
    arms = [x.split("=", 1) for x in a.arm]
    # data[label][suite] = list of (seed, rows)
    data = {}
    for lab, path in arms:
        data[lab] = {}
        for s in SUITES:
            runs = []
            for sd in (SEEDS if "{seed}" in path else [None]):
                r = rows_for(a.root, s, path.replace("{seed}", str(sd)), eps)
                if r:
                    runs.append((sd, r))
            data[lab][s] = runs
    L = [f"# SafeLIBERO scale-up report (episodes {a.eps})", "",
         "Arms: " + "; ".join(f"**{l}** = `{p}`" for l, p in arms), ""]

    def table(title, sel_fn, groups):
        L.extend([f"## {title}", ""])
        for m in MET:
            L.append(f"### {m}")
            L.append("| group | " + " | ".join(l for l, _ in arms) + " |")
            L.append("|---|" + "---|" * len(arms))
            for g in groups:
                cells = []
                for lab, _ in arms:
                    vals, ns = [], []
                    runs_by_seed = {}
                    for s in SUITES:
                        for sd, r in data[lab][s]:
                            runs_by_seed.setdefault(sd, {}).update({(s,) + k: v for k, v in r.items()})
                    for sd, rows in sorted(runs_by_seed.items(), key=lambda x: (x[0] is None, x[0])):
                        k, n = rate(rows, m, lambda key: sel_fn(key, g))
                        if n:
                            vals.append(k / n)
                            ns.append(n)
                    cells.append(fmt_cell(vals, ns) + (f" (n={ns[0]})" if ns and len(vals) == 1 else ""))
                L.append(f"| {g} | " + " | ".join(cells) + " |")
            L.append("")

    table("Overall", lambda k, g: True, ["all 32 scenarios"])
    table("Per suite", lambda k, g: k[0] == g, SUITES)
    table("Per suite x level", lambda k, g: (k[0], k[1]) == tuple(g.split("/")),
          [f"{s}/{l}" for s in SUITES for l in ("I", "II")])
    table("Per task", lambda k, g: (k[0], k[1], str(k[2])) == tuple(g.split("/")),
          [f"{s}/{l}/{t}" for s in SUITES for l in ("I", "II") for t in range(4)])

    # paired tests vs the first arm, per suite, per seed, Holm over suites
    ref = arms[0][0]
    L.extend(["## Paired McNemar vs " + ref, "",
              "b = arm succeeds where the reference fails; c = the reverse. Holm-corrected over the 4 suites.", ""])
    for lab, _ in arms[1:]:
        for m in MET:
            seeds = sorted({sd for s in SUITES for sd, _ in data[lab][s]}, key=lambda x: (x is None, x))
            for sd in seeds:
                ps, info = [], []
                for s in SUITES:
                    refruns = dict(data[ref][s])
                    rr = refruns.get(None, refruns.get(sd))   # a single-run reference pairs with every seed
                    ra = dict(data[lab][s]).get(sd)
                    if rr is None or ra is None:
                        ps.append(1.0)
                        info.append((s, None))
                        continue
                    ks = sorted(set(rr) & set(ra))
                    b = sum(MET[m](ra[k]) and not MET[m](rr[k]) for k in ks)
                    c = sum(MET[m](rr[k]) and not MET[m](ra[k]) for k in ks)
                    p = mcnemar(b, c)
                    ps.append(p)
                    info.append((s, (len(ks), b, c, p)))
                hp = holm(ps)
                parts = []
                for (s, x), h in zip(info, hp):
                    parts.append(f"{s}: missing" if x is None else
                                 f"{s}: n={x[0]} b={x[1]} c={x[2]} p={x[3]:.3g} holm={h:.3g}")
                tag = "" if sd is None else f" seed {sd}"
                L.append(f"- **{lab}{tag}, {m}**: " + "; ".join(parts))
    L.append("")
    # per-seed listing for multi-seed arms
    for lab, path in arms:
        if "{seed}" not in path:
            continue
        L.extend([f"## Per-seed values: {lab}", "", "| suite | seed | n | TSR | CAR | safe |", "|---|---|---|---|---|---|"])
        for s in SUITES:
            for sd, r in data[lab][s]:
                n = len(r)
                L.append(f"| {s} | {sd} | {n} | " + " | ".join(f"{rate(r, m)[0] / n:.3f}" for m in MET) + " |")
        L.append("")
    txt = "\n".join(L)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
