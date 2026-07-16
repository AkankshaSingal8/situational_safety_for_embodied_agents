"""Offline validation of training-free epistemic signals (task #16).

Consumes episode JSONLs that carry per-episode K-candidate telemetry
(`k_disp_mean`, `k_disp_max`, `margin_spread_mean` — emitted by runs after
commit 'epistemic telemetry') and asks: do the signals PREDICT bad outcomes
(collision, timeout-stall) well enough to gate conservative responses?

Reports per-outcome AUC + point-biserial correlation per signal, pooled and
per task. No GPU, no model — pure log analysis.

Usage:
    python analyze_uncertainty_signals.py fgd_composed_fix_n20/results [...more dirs]
"""

import glob
import json
import math
import sys


def auc(pos, neg):
    """Rank-based AUC (Mann-Whitney), ties counted half."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def point_biserial(xs, ys):
    n = len(xs)
    if n < 3 or len(set(ys)) < 2:
        return float("nan")
    mx = sum(xs) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    if sx == 0:
        return float("nan")
    x1 = [x for x, y in zip(xs, ys) if y]
    x0 = [x for x, y in zip(xs, ys) if not y]
    p = len(x1) / n
    return (sum(x1) / len(x1) - sum(x0) / len(x0)) / sx * math.sqrt(p * (1 - p))


def main(dirs):
    eps = []
    for d in dirs:
        for f in glob.glob(f"{d}/**/episodes_*.jsonl", recursive=True):
            for line in open(f):
                e = json.loads(line)
                if "k_disp_mean" in e:
                    e["_src"] = f
                    eps.append(e)
    if not eps:
        print("No episodes with k_disp telemetry found. Run post-telemetry jobs first.")
        return
    step_cap = max(e["steps"] for e in eps)
    for e in eps:
        e["_timeout"] = (not e["success"]) and e["steps"] >= 0.9 * step_cap

    print(f"n={len(eps)} episodes with telemetry (step cap ~{step_cap})")
    signals = ["k_disp_mean", "k_disp_max", "margin_spread_mean"]
    outcomes = [("collision", lambda e: e["collision"]),
                ("timeout_stall", lambda e: e["_timeout"]),
                ("failure_any", lambda e: not e["success"])]
    for oname, ofn in outcomes:
        ys = [bool(ofn(e)) for e in eps]
        base = sum(ys) / len(ys)
        print(f"\n== outcome: {oname} (base rate {base:.2f})")
        for s in signals:
            xs = [e.get(s, float("nan")) for e in eps]
            pos = [x for x, y in zip(xs, ys) if y]
            neg = [x for x, y in zip(xs, ys) if not y]
            print(f"  {s:22s} AUC={auc(pos, neg):.3f}  r_pb={point_biserial(xs, ys):+.3f}")
    print("\nGate (pre-registered): a signal is USEFUL if AUC >= 0.65 on collision or "
          "timeout_stall pooled; else epistemic gating on that signal is dropped.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["fgd_composed_fix_n20/results", "fgd_composed_fix_sp_n20/results"])
