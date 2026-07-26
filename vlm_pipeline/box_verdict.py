"""Acceptance-box verdict for the enforcement finalization (job 42662560).

Rule (user pre-registered): the enforcement config must beat BOTH references
on BOTH axes on Spatial L1 -> TSR >= 67.0 (baseline) AND CAR >= 28.5 (SOTA
reimpl). Reads the LATEST results json per arm, requires n=200, prints the
verdict. Object t1 is quoted separately (repulsor-strength trade-off row).
"""

import glob
import json
import os

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WT, "enforce_final_n50")
BOX_TSR, BOX_CAR = 67.0, 28.5
ARMS = ["fk_eta004", "fk_eta0045", "fk_eta003", "eta005_only"]


def latest(arm, suite):
    fs = sorted(glob.glob(os.path.join(OUT, arm, suite, "I", "results_*.json")))
    if not fs:
        return None
    with open(fs[-1]) as f:
        d = json.load(f)
    for k in ("overall_TSR", "overall_CAR"):
        if d[k] <= 1.0:  # stored as fraction
            d[k] *= 100.0
    return d, os.path.basename(fs[-1])


winners = []
for arm in ARMS:
    sp = latest(arm, "safelibero_spatial")
    ob = latest(arm, "safelibero_object")
    if sp is None:
        print(f"{arm:12s}  spatial: MISSING")
        continue
    d, fn = sp
    tsr, car, n = d["overall_TSR"], d["overall_CAR"], d["total_episodes"]
    full = n >= 200
    in_box = tsr >= BOX_TSR and car >= BOX_CAR
    tag = "IN-BOX" if in_box else "out"
    tag += "" if full else f" (n={n} PILOT ONLY)"
    line = f"{arm:12s}  spatial {tsr:5.1f}/{car:5.1f} n={n:<4d} {tag:24s} [{fn}]"
    if ob is not None:
        od = ob[0]
        line += (f"  | object-t1 {od['overall_TSR']:.1f}/"
                 f"{od['overall_CAR']:.1f} n={od['total_episodes']}")
    print(line)
    if in_box and full:
        winners.append((car, tsr, arm))

print()
if winners:
    winners.sort(reverse=True)
    car, tsr, arm = winners[0]
    print(f"VERDICT: {arm} is THE enforcement config ({tsr:.1f}/{car:.1f} at n=200;"
          f" highest CAR among {len(winners)} in-box arm(s)).")
    print("Next: sbatch --export=ALL,ETA=<eta>,N=50 slurm/ls_soft_rerun.slurm"
          " + minimal-GT board arm.")
else:
    print("VERDICT: no arm holds the box at n=200 -> composed repair stays"
          " production (70.0/46.5 no-GT); noise-space family reported as the"
          " tunable TSR<->CAR frontier.")
