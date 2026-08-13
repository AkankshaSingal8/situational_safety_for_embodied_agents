"""Phase D scorer for the FOL ablation campaign (plan refactored-sprouting-puffin).

Pairs each ablation arm's episodes 1:1 by (level, task, ep) against the frozen
incumbent noGT board cell for the same suite, recomputes TSR / viol% / safe-SR
from the episode JSONLs (never trusts results_*.json), and runs exact McNemar
on success and on violation.

Incumbent provenance mirrors the ledgered TABLE 6 assembly
(results_tables/fgd_tier_gt_results.md, 2026-08-08):
  oa  -> ls_ours_parity_full/nogt_par_full   (fol, eng0.30)
  oah -> ls_cone_confirm/F3_oah_nogt_cone    (fol+cone60)
  hs  -> ls_topup/hsL0_nogt_folprop L0 + ls_gapfix_dev/CD_hs_nogt_folprop L1/L2
  aff -> ls_aff_fpvlm_cone                   (folpropvlm+cone60)

Ablation arms are read from ls_fol_ablate/A_<suite-short>_<ablation>/<suite>/L<l>.
Arms with no episodes yet are reported as PENDING, so the script can run at any
point during the campaign.
"""

import argparse
import json
import pathlib

from stats_tests import mcnemar_exact

ROOT = pathlib.Path(__file__).parents[1]

SUITES = {
    "oa": "obstacle_avoidance",
    "oah": "obstacle_avoidance_human",
    "hs": "human_safety",
    "aff": "affordance",
}

# level -> episodes dir of the frozen incumbent noGT cell
INCUMBENT = {
    "oa": {l: ROOT / "ls_ours_parity_full/nogt_par_full/obstacle_avoidance" / f"L{l}"
           for l in (0, 1, 2)},
    "oah": {l: ROOT / "ls_cone_confirm/F3_oah_nogt_cone/obstacle_avoidance_human" / f"L{l}"
            for l in (0, 1, 2)},
    "hs": {0: ROOT / "ls_topup/hsL0_nogt_folprop/human_safety/L0",
           1: ROOT / "ls_gapfix_dev/CD_hs_nogt_folprop/human_safety/L1",
           2: ROOT / "ls_gapfix_dev/CD_hs_nogt_folprop/human_safety/L2"},
    "aff": {l: ROOT / "ls_aff_fpvlm_cone/affordance" / f"L{l}" for l in (0, 1, 2)},
}

ARMS = [  # (suite-short, ablation) — the 9 controller-fixed rollout cells
    ("oa", "geom_only"), ("oa", "no_exempt"),
    ("oah", "geom_only"), ("oah", "no_exempt"), ("oah", "no_class"),
    ("hs", "geom_only"), ("hs", "no_class"),
    ("aff", "geom_only"), ("aff", "no_exempt"),
]


def load_cell(cell_dir):
    """(task, ep) -> record, dedup last-wins (logs append across reruns)."""
    files = sorted(pathlib.Path(cell_dir).glob("episodes_*.jsonl"))
    eps = {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if not {"task", "ep", "success", "violation"} <= r.keys():
                raise ValueError(f"corrupt record in {f}: {sorted(r.keys())}")
            eps[(r["task"], r["ep"])] = r
    return eps


def pooled(eps):
    n = len(eps)
    s = sum(1 for r in eps.values() if r["success"])
    v = sum(1 for r in eps.values() if r["violation"])
    ss = sum(1 for r in eps.values() if r["success"] and not r["violation"])
    return {"n": n, "tsr": 100 * s / n, "viol": 100 * v / n, "safe_sr": 100 * ss / n}


def score_arm(suite_short, ablation, ablate_root):
    suite = SUITES[suite_short]
    arm_dir = ablate_root / f"A_{suite_short}_{ablation}" / suite
    inc_all, abl_all = {}, {}
    levels_done = []
    for lvl, inc_dir in INCUMBENT[suite_short].items():
        abl_dir = arm_dir / f"L{lvl}"
        abl = load_cell(abl_dir) if abl_dir.is_dir() else {}
        if not abl:
            continue
        inc = load_cell(inc_dir)
        common = set(inc) & set(abl)
        if len(common) != len(abl):
            print(f"  WARN {suite_short}/{ablation} L{lvl}: "
                  f"{len(abl) - len(common)} ablation eps lack incumbent pair")
        for k in common:
            inc_all[(lvl,) + k] = inc[k]
            abl_all[(lvl,) + k] = abl[k]
        levels_done.append(lvl)
    if not abl_all:
        return {"arm": f"{suite_short}/{ablation}", "status": "PENDING"}

    res = {"arm": f"{suite_short}/{ablation}", "status":
           "COMPLETE" if levels_done == [0, 1, 2] else f"PARTIAL L{levels_done}",
           "incumbent": pooled(inc_all), "ablated": pooled(abl_all)}
    for metric in ("success", "violation"):
        b = sum(1 for k in abl_all if inc_all[k][metric] and not abl_all[k][metric])
        c = sum(1 for k in abl_all if not inc_all[k][metric] and abl_all[k][metric])
        res[f"mcnemar_{metric}"] = {"inc_only": b, "abl_only": c,
                                    "p": mcnemar_exact(b, c)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate_root", default=str(ROOT / "ls_fol_ablate"))
    ap.add_argument("--selfcheck", action="store_true",
                    help="pair each incumbent against itself (expect delta 0, p=1)")
    args = ap.parse_args()

    if args.selfcheck:
        for ss in SUITES:
            inc = {}
            for lvl, d in INCUMBENT[ss].items():
                for k, r in load_cell(d).items():
                    inc[(lvl,) + k] = r
            p = pooled(inc)
            b = c = 0
            assert mcnemar_exact(b, c) == 1.0
            print(f"selfcheck {ss}: n={p['n']} tsr={p['tsr']:.1f} "
                  f"viol={p['viol']:.1f} safe_sr={p['safe_sr']:.1f} (self-pair p=1.0 ok)")
        return

    out = [score_arm(ss, ab, pathlib.Path(args.ablate_root)) for ss, ab in ARMS]
    for r in out:
        if r["status"] == "PENDING":
            print(f"{r['arm']:<16} PENDING")
            continue
        i, a = r["incumbent"], r["ablated"]
        ms, mv = r["mcnemar_success"], r["mcnemar_violation"]
        print(f"{r['arm']:<16} {r['status']:<10} n={a['n']:>3}  "
              f"TSR {i['tsr']:5.1f} -> {a['tsr']:5.1f}  "
              f"viol {i['viol']:4.1f} -> {a['viol']:4.1f}  "
              f"safeSR {i['safe_sr']:5.1f} -> {a['safe_sr']:5.1f}  "
              f"pS={ms['p']:.3g} ({ms['inc_only']}:{ms['abl_only']})  "
              f"pV={mv['p']:.3g} ({mv['inc_only']}:{mv['abl_only']})")
    print(json.dumps(out, indent=1), file=open(
        ROOT / "results_tables" / "fol_ablation_scores.json", "w"))
    print("\nwrote results_tables/fol_ablation_scores.json")


if __name__ == "__main__":
    main()
