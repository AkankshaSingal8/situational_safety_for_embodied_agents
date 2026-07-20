"""Assemble the paper's Table 1 from results JSONs + Fisher tests.

Rows: pi0.5 baseline, SOTA-reimpl, AEGIS self-run, ours GT, ours no-GT
(VLM-general). Static rows carry the ledgered n=50 finals; live rows (AEGIS,
no-GT VLM-general) are read from results dirs as cells land — missing cells
print as '…'.

Bold rule (pre-registered): beats BOTH baseline AND SOTA-reimpl on BOTH axes.
Fisher exact per cell vs both references on both axes (n=200 successes /
collision-free counts).
"""

import json
import pathlib
import sys

from scipy.stats import fisher_exact

WT = pathlib.Path(__file__).parents[1]
CELLS = [("safelibero_spatial", "I"), ("safelibero_spatial", "II"),
         ("safelibero_goal", "I"), ("safelibero_goal", "II"),
         ("safelibero_object", "I"), ("safelibero_object", "II"),
         ("safelibero_long", "I"), ("safelibero_long", "II")]

BASELINE = {("safelibero_spatial", "I"): (67.0, 14.0), ("safelibero_spatial", "II"): (55.5, 12.0),
            ("safelibero_goal", "I"): (51.0, 23.0), ("safelibero_goal", "II"): (66.5, 35.0),
            ("safelibero_object", "I"): (40.5, 14.0), ("safelibero_object", "II"): (74.0, 25.0),
            ("safelibero_long", "I"): (58.0, 15.0), ("safelibero_long", "II"): (51.0, 16.5)}
REIMPL = {("safelibero_spatial", "I"): (62.0, 28.5), ("safelibero_spatial", "II"): (71.5, 25.5),
          ("safelibero_goal", "I"): (71.0, 27.5), ("safelibero_goal", "II"): (75.0, 49.5),
          ("safelibero_object", "I"): (53.0, 15.5), ("safelibero_object", "II"): (76.0, 33.5),
          ("safelibero_long", "I"): (30.5, 35.5), ("safelibero_long", "II"): (43.0, 59.5)}
OURS_GT = {("safelibero_spatial", "I"): (71.0, 23.5), ("safelibero_spatial", "II"): (85.0, 76.5),
           ("safelibero_goal", "I"): (77.5, 51.0), ("safelibero_goal", "II"): (76.0, 60.0),
           ("safelibero_object", "I"): (53.5, 28.0), ("safelibero_object", "II"): (83.0, 74.5),
           ("safelibero_long", "I"): (42.5, 71.0), ("safelibero_long", "II"): (47.0, 77.0)}
# No-GT VLM-general: Spatial L1 landed 2026-07-19 (job 42430870); rest stream in.
NOGT_DONE = {("safelibero_spatial", "I"):
             str(WT / "fgd_nogt_spatial_vlm/results/safelibero_spatial/I")}
# Hand-table no-GT board, kept for the prior-source ablation column.
NOGT_TABLE = {("safelibero_spatial", "I"): (60.0, 40.0), ("safelibero_spatial", "II"): (87.0, 68.5),
              ("safelibero_goal", "I"): (77.5, 67.5), ("safelibero_goal", "II"): (74.0, 68.0),
              ("safelibero_object", "I"): (56.5, 80.5), ("safelibero_object", "II"): (77.0, 86.0),
              ("safelibero_long", "I"): (49.0, 66.5), ("safelibero_long", "II"): (44.0, 79.5)}


def read_cell(root):
    root = pathlib.Path(root)
    files = sorted(root.rglob("results_*.json"))
    if not files:
        return None
    r = json.load(open(files[-1]))
    if "overall_TSR" in r:
        return (100 * r["overall_TSR"], 100 * r["overall_CAR"], r["total_episodes"])
    # AEGIS schema
    return (100 * r["TSR"], 100 * r["CAR"], r.get("total_episodes", 200))


def fisher(ours, ref, n=200):
    a = round(ours / 100 * n)
    b = round(ref / 100 * n)
    return fisher_exact([[a, n - a], [b, n - b]])[1]


def fmt(cell, key):
    if cell is None:
        return "…"
    tsr, car = cell[0], cell[1]
    base, re = BASELINE[key], REIMPL[key]
    win = tsr > base[0] and tsr > re[0] and car > base[1] and car > re[1]
    s = f"{tsr:.1f}/{car:.1f}"
    return f"**{s}**" if win else s


def main():
    print("| Condition | pi0.5 | SOTA-reimpl | AEGIS | Ours GT | Ours no-GT (VLM-general) |")
    print("|---|---|---|---|---|---|")
    for key in CELLS:
        suite, lvl = key
        aegis = read_cell(WT / f"fgd_aegis_selfrun/{suite}")  # results_<level>.json layout differs
        af = WT / f"fgd_aegis_selfrun/{suite}/results_{lvl}.json"
        aegis = None
        if af.exists():
            r = json.load(open(af))
            aegis = (100 * r["TSR"], 100 * r["CAR"], r.get("total_episodes", 200))
        nogt_dir = NOGT_DONE.get(key, WT / f"fgd_vlmfin/{suite}_{lvl}")
        nogt = read_cell(nogt_dir)
        name = suite.replace("safelibero_", "").capitalize()
        print(f"| {name} L{ '1' if lvl=='I' else '2'} "
              f"| {BASELINE[key][0]}/{BASELINE[key][1]} | {REIMPL[key][0]}/{REIMPL[key][1]} "
              f"| {fmt(aegis, key)} | {fmt(OURS_GT[key], key)} | {fmt(nogt, key)} |")
        if nogt:
            pt_b = fisher(nogt[0], BASELINE[key][0]); pc_b = fisher(nogt[1], BASELINE[key][1])
            pt_r = fisher(nogt[0], REIMPL[key][0]); pc_r = fisher(nogt[1], REIMPL[key][1])
            tab = NOGT_TABLE[key]
            print(f"|  |  |  |  |  | vs base p_T={pt_b:.2g} p_C={pc_b:.2g}; "
                  f"vs reimpl p_T={pt_r:.2g} p_C={pc_r:.2g}; table-arm {tab[0]}/{tab[1]} |")


if __name__ == "__main__":
    main()
