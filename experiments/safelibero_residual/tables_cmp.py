"""Tables 1a (layouts 0-39), 1b (40-49), 2 (25 novel objects), 3 (new obstacle positions): pi0.5 vs arms.
    python tables_cmp.py ex3b_bs64/true ex7last/true      cells: TSR / CAR / SR, then SR gain per arm"""
import sys
from make_tables import SUITES, cell, load
arms = sys.argv[1:]
T = [("Table 1a: SafeLIBERO, layouts 0-39", "test", ["eps0-19", "eps20-39"]),
     ("Table 1b: SafeLIBERO, layouts 40-49", "test", ["eps40-49"]),
     ("Table 2: 25 novel objects", "scenes_newobj25", ["all"]),
     ("Table 3: new obstacle positions", "scenes_newpos", ["all"])]
def sr(c): return float(c[0].split("/")[-1]) if c[1] else None
for title, sub, eps in T:
    print(f"\n### {title}\n| suite | lvl | n | pi0.5 | " + " | ".join(arms) + " | " + " | ".join(f"gain {a.split('/')[0]}" for a in arms) + " |")
    for s in SUITES:
        data = {a: sum((load(f"runs_v2/{s}/{sub}/{d}/{e}/part_*.jsonl") for e in eps), [])
                for a, d in [("pi05", "pi05")] + [(a, a) for a in arms]}
        for L in ("I", "II"):
            c = {a: cell([r for r in v if r["level"] == L]) for a, v in data.items()}
            g = [f"{sr(c[a]) - sr(c['pi05']):+.1f}" if sr(c[a]) is not None else "-" for a in arms]
            print(f"| {s} | {L} | {c['pi05'][1]} | {c['pi05'][0]} | " + " | ".join(f"{c[a][0]} (n={c[a][1]})" for a in arms) + " | " + " | ".join(g) + " |")
