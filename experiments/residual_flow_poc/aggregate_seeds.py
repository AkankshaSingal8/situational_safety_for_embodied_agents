"""Combine results_v2.json from several training seeds.
    python aggregate_seeds.py --runs runs_v2 runs_v2_s1 runs_v2_s2 --out runs_v2/seeds
writes <out>/seeds.json and <out>/seeds.md: per-seed value, mean and sample s.d. across seeds,
for every headline metric on train and test scenes.
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", default="runs_v2/seeds")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

RS = {r: json.load(open(os.path.join(r, "results_v2.json"))) for r in a.runs}
# training seed from the directory name: runs_v2 is seed 0, runs_v2_s<N> is seed N
LABELS = [f"seed {r.rstrip('/').rsplit('_s', 1)[1]}" if "_s" in os.path.basename(r.rstrip('/')) else "seed 0" for r in a.runs]


def p(x):
    return x["k"] / x["n"]


METRICS = [
    ("Exp1 A text: laptop avoided & placed", lambda R: p(R["summary"]["main|A"]["A_ok&succ"])),
    ("Exp1 A text: tilt still violated (should be 1)", lambda R: p(R["summary"]["main|A"]["viol_B"])),
    ("Exp1 B text: upright & placed", lambda R: p(R["summary"]["main|B"]["B_ok&succ"])),
    ("Exp1 B text: laptop still violated (should be 1)", lambda R: p(R["summary"]["main|B"]["viol_A"])),
    ("Exp1 no-text model: laptop avoided & placed", lambda R: p(R["summary"]["notext"]["A_ok&succ"])),
    ("Exp1 no-text model: upright & placed", lambda R: p(R["summary"]["notext"]["B_ok&succ"])),
    ("Exp1 shuffled, A text: laptop avoided & placed", lambda R: p(R["summary"]["shuffled|A"]["A_ok&succ"])),
    ("Exp1 shuffled, B text: upright & placed", lambda R: p(R["summary"]["shuffled|B"]["B_ok&succ"])),
    ("Exp1 detour on the reference side", lambda R: R["detour_side_agreement"]),
    ("Exp1 main_ab A text: laptop avoided & placed", lambda R: p(R["summary"]["main_ab|A"]["A_ok&succ"])),
    ("Exp1 main_ab B text: upright & placed", lambda R: p(R["summary"]["main_ab|B"]["B_ok&succ"])),
    ("Exp2 unseen laptop wordings, set 1", lambda R: p(R["exp2_heldout_A"])),
    ("Exp2 unseen laptop wordings, set 2", lambda R: p(R["exp2_heldout2_A"])),
    ("Exp2 unseen upright wordings, set 1", lambda R: p(R["exp2_heldout_B"])),
    ("Exp2 unseen upright wordings, set 2", lambda R: p(R["exp2_heldout2_B"])),
]
for pre, lab in [("main|combined", "close set: sentence given directly"),
                 ("compose|combined", "close set: velocity composition"),
                 ("stack_AB|combined", "close set: clauses in turn A then B"),
                 ("stack_BA|combined", "close set: clauses in turn B then A"),
                 ("sum|combined", "close set: corrections added"),
                 ("main_ab|combined", "close set: trained on combined sentences"),
                 ("main_ab|trainAB", "main_ab on its own training sentences"),
                 ("main|far", "far set: sentence given directly"),
                 ("stack_AB|far", "far set: clauses in turn A then B"),
                 ("stack_BA|far", "far set: clauses in turn B then A"),
                 ("sum|far", "far set: corrections added"),
                 ("main_ab|far", "far set: trained on combined sentences")]:
    METRICS.append((f"Exp3 {lab}", lambda R, pre=pre: p(R["exp3"][pre]["AB_ok&succ"])))
METRICS.append(("Exp3 far set: corrections added, tilt violated", lambda R: p(R["exp3"]["sum|far"]["viol_B"])))
METRICS.append(("Exp3 far set: clauses in turn A then B, tilt violated", lambda R: p(R["exp3"]["stack_AB|far"]["viol_B"])))

out, md = {}, ["# Residual flow v2 across training seeds\n",
               f"Runs: {', '.join(a.runs)}. Each cell: per-seed values, then mean ± sample s.d. across seeds.\n"]
for split in ("train", "test"):
    md += [f"\n## {split} scenes\n", "| metric | " + " | ".join(LABELS) + " | mean ± s.d. |",
           "|---|" + "---|" * (len(RS) + 1)]
    out[split] = {}
    for name, f in METRICS:
        vals = [f(R[split]) for R in RS.values()]
        m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[split][name] = dict(values=vals, mean=m, sd=sd)
        md.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {m:.3f} ± {sd:.3f} |")
# paired tests per seed (test split)
md += ["\n## Paired comparisons per seed (test scenes, sample 0; b = first arm better, c = second)\n",
       "| comparison | " + " | ".join(f"{l} (b/c, p)" for l in LABELS) + " |", "|---|" + "---|" * len(RS)]
keys = sorted(set.intersection(*[set(R["test"]["tests"]) for R in RS.values()]))
for k in keys:
    md.append(f"| {k} | " + " | ".join(f"{R['test']['tests'][k]['b']}/{R['test']['tests'][k]['c']}, {R['test']['tests'][k]['p']:.2g}"
                                         for R in RS.values()) + " |")
json.dump(out, open(os.path.join(a.out, "seeds.json"), "w"), indent=1)
open(os.path.join(a.out, "seeds.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
