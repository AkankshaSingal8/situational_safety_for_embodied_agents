"""Closed-loop validation (checkpoint / scale selection): SR gain over pi0.5 on scenes_val (48 val GSO objects)
and scenes_val_sl (fresh SafeLIBERO placements, layouts 0-11).  python val_tables.py ARMDIR [ARMDIR ...]"""
import sys

from make_tables import SUITES, cell, load

arms = [("pi0.5", "pi05")] + [(d, d) for d in sys.argv[1:]]
for sub in ["scenes_val", "scenes_val_sl"]:
    print(f"### {sub}\n| suite | lvl | n | " + " | ".join(a for a, _ in arms) + " |")
    for s in SUITES:
        data = {a: load(f"runs_v2/{s}/{sub}/{d}/*/part_*.jsonl") for a, d in arms}
        for L in ("I", "II"):
            row = [cell([r for r in data[a] if r["level"] == L]) for a, _ in arms]
            print(f"| {s} | {L} | {row[0][1]} | " + " | ".join(t for t, _ in row) + " |")
