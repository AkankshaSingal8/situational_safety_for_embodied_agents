"""Tables 1-3 for one arm vs pi0.5: TSR / CAR / SR and SR gain.  python tables_best.py ex3b_bs64/true"""
import sys
from make_tables import SUITES, cell, load
d = sys.argv[1]
T = [("Table 1 (official test, SafeLIBERO obstacles)", "test", "*"),
     ("Table 2 (25 novel objects)", "scenes_newobj25", "all"),
     ("Table 3 (new obstacle positions)", "scenes_newpos", "all")]
for title, sub, eps in T:
    print(f"\n### {title}\n| suite | lvl | n | pi0.5 TSR/CAR/SR | {d} TSR/CAR/SR | SR gain |\n|---|---|---|---|---|---|")
    for s in SUITES:
        b = load(f"runs_v2/{s}/{sub}/pi05/{eps}/part_*.jsonl")
        e = load(f"runs_v2/{s}/{sub}/{d}/{eps}/part_*.jsonl")
        for L in ("I", "II"):
            cb, ce = cell([r for r in b if r["level"] == L]), cell([r for r in e if r["level"] == L])
            try:
                g = f"{float(ce[0].split('/')[-1]) - float(cb[0].split('/')[-1]):+.1f}"
            except ValueError:
                g = "-"
            print(f"| {s} | {L} | {cb[1]} | {cb[0]} | {ce[0]} | {g} |")
