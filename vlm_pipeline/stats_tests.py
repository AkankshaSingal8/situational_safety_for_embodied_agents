"""Significance tests for TSR/CAR claims (paper statistics discipline).

Two modes:
1. fisher: unpaired two-proportion Fisher exact test from aggregate counts
   (ours vs baseline — baseline runs predate episode-level logging).
2. mcnemar: paired McNemar exact test between two runs' episode JSONLs,
   paired on (task, ep) — valid because all runs share seed 7 + official
   init states.

Usage:
    python stats_tests.py fisher --a-succ 170 --a-n 200 --b-succ 111 --b-n 200
    python stats_tests.py mcnemar runA_episodes.jsonl runB_episodes.jsonl --metric success
"""

import argparse
import json
import math


def _log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact_two_sided(a, b, c, d):
    """2x2 table [[a,b],[c,d]] — two-sided by summing tables with p <= p_obs."""
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def p_table(x):
        return math.exp(_log_comb(col1, x) + _log_comb(n - col1, row1 - x) - _log_comb(n, row1))

    lo = max(0, row1 + col1 - n)
    hi = min(row1, col1)
    p_obs = p_table(a)
    return sum(p for p in (p_table(x) for x in range(lo, hi + 1)) if p <= p_obs * (1 + 1e-9))


def mcnemar_exact(b, c):
    """Exact binomial McNemar on discordant pairs (b: A-only success, c: B-only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.exp(_log_comb(n, i)) * 0.5 ** n for i in range(0, k + 1))
    return min(1.0, 2 * p)


def load_eps(path):
    out = {}
    for line in open(path):
        e = json.loads(line)
        out[(e["task"], e["ep"])] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("fisher")
    f.add_argument("--a-succ", type=int, required=True)
    f.add_argument("--a-n", type=int, required=True)
    f.add_argument("--b-succ", type=int, required=True)
    f.add_argument("--b-n", type=int, required=True)
    m = sub.add_parser("mcnemar")
    m.add_argument("run_a")
    m.add_argument("run_b")
    m.add_argument("--metric", choices=["success", "collision"], default="success")
    args = ap.parse_args()

    if args.mode == "fisher":
        a, b = args.a_succ, args.a_n - args.a_succ
        c, d = args.b_succ, args.b_n - args.b_succ
        p = fisher_exact_two_sided(a, b, c, d)
        print(f"A: {a}/{args.a_n} ({a/args.a_n:.1%})  B: {c}/{args.b_n} ({c/args.b_n:.1%})")
        print(f"Fisher exact two-sided p = {p:.2e}")
    else:
        ea, eb = load_eps(args.run_a), load_eps(args.run_b)
        keys = sorted(set(ea) & set(eb))
        if len(keys) < len(ea) or len(keys) < len(eb):
            print(f"warning: pairing on {len(keys)} common episodes "
                  f"(A has {len(ea)}, B has {len(eb)})")
        b_cnt = sum(bool(ea[k][args.metric]) and not eb[k][args.metric] for k in keys)
        c_cnt = sum(not ea[k][args.metric] and bool(eb[k][args.metric]) for k in keys)
        both = sum(bool(ea[k][args.metric]) and bool(eb[k][args.metric]) for k in keys)
        p = mcnemar_exact(b_cnt, c_cnt)
        print(f"paired n={len(keys)}  {args.metric}: A-only={b_cnt}  B-only={c_cnt}  both={both}")
        print(f"McNemar exact p = {p:.2e}")


if __name__ == "__main__":
    main()
