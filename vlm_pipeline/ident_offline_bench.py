"""Offline identity bench + calibration sweep over captured SafeLIBERO scenes.

Scores hazard-identity functions against results_tables/safelibero_ident_scenes.jsonl
(produced by ident_scene_capture.py). Baselines are the EXACT client paths:

  symbolic        scored_obstacle_id via the identify_obstacle candidate view
                  (record 'objects' = the same workspace-filtered _cands),
                  hand HAZARD_PRIOR table
  symbolic+vlm    same, after load_vlm_priors(results_tables/vlm_hazard_priors.json,
                  floor=0.5) -- mirrors run_guided --hazard_prior_source vlm defaults
  fol             fol_obstacle_id top-1 on the client's fol entity view
                  (record 'objects_all' = NO workspace filter, exactly the
                  eval client's fol _cands construction)

Reproduction gate (documented board numbers, 50-ep runs; ours are the first
10 init states): goal II t1 ~= 0.18 symbolic / 0.00 fol, t2 ~= 0.78 / 0.06,
t0 ~= 0.94, t3 ~= 1.0.

Calibration sweep: rule structure unchanged; only HOW the hazard prior enters
the v3 score is varied --
  mult:  score = prior^alpha * exp(-d^2/2s^2) * (1 - 0.8*frac)   (alpha=1 = current)
  add:   score = (beta*prior + (1-beta)*path) * (1 - 0.8*frac)
  veto:  candidates with prior < tau are ineligible for the near-path argmax
         unless PROTECTED/MOVING (fallback to all if everything is vetoed)
Objective: max pooled accuracy s.t. every currently-healthy cell (baseline
acc >= 0.85) drops by at most 0.05.

Usage:
  python vlm_pipeline/ident_offline_bench.py [--scenes ...] [--sweep]
"""

import argparse
import itertools
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import symbolic_identity as SI  # noqa: E402
from symbolic_identity import (  # noqa: E402
    _PROTECTED_CLASSES,
    _PROTECTED_PATH_FLOOR,
    _clean_object_name,
    fol_obstacle_id,
    scored_obstacle_id,
)

REPO = pathlib.Path(__file__).parents[1]


# ---------------------------------------------------------------- baselines

def set_priors(source, vlm_path, floor=0.5):
    """source: 'table' (hand HAZARD_PRIOR) or 'vlm' (mirrors client load)."""
    if source == "table":
        SI._VLM_PRIORS = None
    else:
        SI.load_vlm_priors(str(vlm_path), floor=floor)


def run_symbolic(rec):
    cands = {k: np.asarray(v) for k, v in rec["objects"].items()}
    if not cands:
        return None
    return scored_obstacle_id(rec["task_description"], cands,
                              np.asarray(rec["eef_pos"]))


def run_fol(rec):
    cands = {k: np.asarray(v) for k, v in rec["objects_all"].items()}
    if not cands:
        return None
    ranked = fol_obstacle_id(rec["task_description"], cands,
                             np.asarray(rec["eef_pos"]))
    return ranked[0] if ranked else None


# ------------------------------------------------- parametric v3 scorer
# Verbatim replica of symbolic_identity.scored_obstacle_id with the prior-entry
# knobs exposed (mode/alpha/beta/tau). Structure (mention_frac, head-noun rule,
# protected override, multi-goal path term) is UNCHANGED.

def vlm_token_weight(name, priors, floor, default=0.4):
    """VLM-prior lookup with token/substring fallback (mirrors the hand
    table's token-max matching): 'moka pot small' -> 'moka pot' rating
    instead of the unseen-name default. No GT knowledge consulted."""
    clean = _clean_object_name(name).lower().replace(" obstacle", "")
    rated = priors.get(clean)
    if rated is None:
        hits = [v for k, v in priors.items() if k in clean or clean in k]
        rated = max(hits) if hits else default
    return max(rated, floor)


def scored_variant(task_description, candidate_positions, eef_pos,
                   sigma=0.25, mode="mult", alpha=1.0, beta=0.5, tau=0.0,
                   moving=None, weight_fn=None):
    task = task_description.lower()

    def mention_frac(key):
        toks = [w for w in _clean_object_name(key).lower().split()
                if len(w) >= 3 and w != "obstacle"]
        if not toks:
            return 0.0
        frac = sum(w in task for w in toks) / len(toks)
        if toks[-1] in task:
            frac = 1.0
        return frac

    fracs = {k: mention_frac(k) for k in candidate_positions}
    protected = {k for k in candidate_positions
                 if any(w in _clean_object_name(k).lower()
                        for w in _PROTECTED_CLASSES)}
    if moving:
        protected = protected | {k for k in candidate_positions if k in moving}
    for k in protected:
        fracs[k] = 0.0
    partial = {k for k, f in fracs.items() if f < 1.0}
    if not partial:
        return None
    goals = [k for k, f in fracs.items() if f >= 0.5]

    spos = np.asarray(eef_pos)[:2] if eef_pos is not None else np.array([-0.21, 0.0])
    segs = [(spos, np.asarray(candidate_positions[g])[:2]) for g in goals]
    if not segs:
        segs.append((spos, np.array([0.0, 0.15])))

    def d_path(k):
        p = np.asarray(candidate_positions[k])[:2]
        best = np.inf
        for s, e in segs:
            seg = e - s
            l2 = float(seg @ seg)
            t = 0.0 if l2 < 1e-9 else float(np.clip((p - s) @ seg / l2, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(p - (s + t * seg))))
        return best

    wfn = weight_fn if weight_fn is not None else SI._hazard_weight

    def score(k):
        path = float(np.exp(-d_path(k) ** 2 / (2 * sigma ** 2)))
        w = wfn(k)
        if k in protected:
            path = max(path, _PROTECTED_PATH_FLOOR)
            w = 1.0
        if mode == "mult":
            prior_term = w ** alpha
            return prior_term * path * (1.0 - 0.8 * fracs[k])
        # additive blend
        return (beta * w + (1.0 - beta) * path) * (1.0 - 0.8 * fracs[k])

    # Low-prior veto: benign-class objects may not be selected by the
    # near-path rule (they remain goal/support entities); PROTECTED/MOVING
    # are exempt. Fail-safe fallback: if the veto empties the pool, ignore it.
    eligible = {k for k in partial
                if k in protected or wfn(k) >= tau}
    pool = eligible if eligible else partial
    return max(pool, key=score)


# ---------------------------------------------------------------- scoring

def cell_key(rec):
    return (rec["suite"], rec["level"], rec["task_id"])


def evaluate(records, fn):
    """fn: rec -> picked. Returns (per_cell_acc, pooled_acc, per_cell_n)."""
    hits, ns = defaultdict(int), defaultdict(int)
    for rec in records:
        if rec["gt_obstacle_name"] is None:
            continue
        picked = fn(rec)
        ns[cell_key(rec)] += 1
        hits[cell_key(rec)] += int(picked == rec["gt_obstacle_name"])
    per_cell = {c: hits[c] / ns[c] for c in ns}
    pooled = sum(hits.values()) / max(sum(ns.values()), 1)
    return per_cell, pooled, dict(ns)


def print_table(name, per_cell, pooled):
    print(f"\n--- {name} (pooled {pooled:.3f}) ---")
    for c in sorted(per_cell):
        print(f"  {c[0]:<20} L{c[1]} t{c[2]}: {per_cell[c]:.2f}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=str,
                    default=str(REPO / "results_tables/safelibero_ident_scenes.jsonl"))
    ap.add_argument("--vlm_priors", type=str,
                    default=str(REPO / "results_tables/vlm_hazard_priors.json"))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--healthy_thresh", type=float, default=0.85)
    ap.add_argument("--max_regress", type=float, default=0.05)
    args = ap.parse_args()

    records = [json.loads(line) for line in open(args.scenes)]
    n_gt = sum(r["gt_obstacle_name"] is not None for r in records)
    print(f"{len(records)} records ({n_gt} with GT hazard), "
          f"{len({cell_key(r) for r in records})} cells")

    baselines = {}
    set_priors("table", args.vlm_priors)
    baselines["symbolic(table)"] = evaluate(records, run_symbolic)
    set_priors("vlm", args.vlm_priors, floor=0.5)
    baselines["symbolic(vlm f=0.5)"] = evaluate(records, run_symbolic)
    set_priors("table", args.vlm_priors)
    baselines["fol top-1"] = evaluate(records, run_fol)

    for name, (pc, pooled, _) in baselines.items():
        print_table(name, pc, pooled)

    # Reproduction gate vs board numbers (50-ep runs; ours = 10 init states).
    print("\n=== reproduction gate (safelibero_goal level II) ===")
    expect = {("symbolic", 0): 0.94, ("symbolic", 1): 0.18,
              ("symbolic", 2): 0.78, ("symbolic", 3): 1.0,
              ("fol", 1): 0.0, ("fol", 2): 0.06}
    for (arm, t), exp in sorted(expect.items()):
        for bname in baselines:
            if (arm == "symbolic" and bname.startswith("symbolic")) or \
               (arm == "fol" and bname.startswith("fol")):
                got = baselines[bname][0].get(("safelibero_goal", "II", t))
                print(f"  goal II t{t} {bname:<20}: bench={got if got is None else round(got, 2)}"
                      f"  board~{exp}")

    if not args.sweep:
        return

    # ------------------------------------------------------------ sweep
    # Two anchors, each with the no-regression constraint measured against its
    # OWN currently-deployed config: (a) table priors (client default), (b)
    # VLM priors floor=0.5 (--hazard_prior_source vlm default -- the config
    # the no-GT board rows were run with, and the one that reproduces the
    # goal II failures). For the vlm anchor the protection floor is itself a
    # swept knob (floor interactions were named in the diagnosis).
    def sweep_grid(with_floor):
        grids = []
        floors = [0.5, 0.3, 0.0] if with_floor else [None]
        lookups = ["exact", "token"] if with_floor else ["exact"]
        for lookup in lookups:
            for floor in floors:
                for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
                    for tau in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]:
                        for sigma in [0.2, 0.25, 0.35]:
                            grids.append((floor, lookup, dict(mode="mult", alpha=alpha, tau=tau, sigma=sigma)))
                for beta in [0.3, 0.5, 0.7]:
                    for tau in [0.0, 0.3, 0.5]:
                        for sigma in [0.2, 0.25, 0.35]:
                            grids.append((floor, lookup, dict(mode="add", beta=beta, tau=tau, sigma=sigma)))
        return grids

    raw_priors = {k.lower(): float(v)
                  for k, v in json.load(open(args.vlm_priors)).items()}

    for source, cur_floor in [("table", None), ("vlm", 0.5)]:
        anchor = source if source == "table" else f"vlm floor={cur_floor}"
        print(f"\n=== calibration sweep, anchor = symbolic({anchor}) current params ===")
        set_priors(source, args.vlm_priors, floor=cur_floor or 0.5)
        base_pc, base_pooled, _ = evaluate(
            records, lambda r: scored_variant(
                r["task_description"], {k: np.asarray(v) for k, v in r["objects"].items()},
                np.asarray(r["eef_pos"])))
        healthy = {c for c, a in base_pc.items() if a >= args.healthy_thresh}
        print(f"base pooled {base_pooled:.3f}, {len(healthy)}/{len(base_pc)} healthy cells")

        results = []
        for floor, lookup, g in sweep_grid(with_floor=(source == "vlm")):
            if source == "vlm" and lookup == "token":
                wfn = (lambda n, f=floor: vlm_token_weight(n, raw_priors, f))
            else:
                wfn = None
                set_priors(source, args.vlm_priors, floor=floor if floor is not None else 0.5)
            pc, pooled, _ = evaluate(
                records, lambda r: scored_variant(
                    r["task_description"],
                    {k: np.asarray(v) for k, v in r["objects"].items()},
                    np.asarray(r["eef_pos"]), weight_fn=wfn, **g))
            worst_reg = max((base_pc[c] - pc.get(c, 0.0) for c in healthy), default=0.0)
            ok = worst_reg <= args.max_regress + 1e-9
            results.append((floor, lookup, g, pooled, worst_reg, ok, pc))

        feasible = sorted([r for r in results if r[5]], key=lambda r: (-r[3], r[4]))
        print(f"{len(feasible)}/{len(results)} configs satisfy no-regression; top 10:")
        for floor, lookup, g, pooled, worst_reg, _, pc in feasible[:10]:
            ftag = f"floor={floor} lookup={lookup} " if floor is not None else ""
            print(f"  {ftag}{g}  pooled={pooled:.3f}  worst_healthy_reg={worst_reg:.2f}")

        if feasible:
            floor, lookup, g, pooled, worst_reg, _, pc = feasible[0]
            ftag = f"floor={floor} lookup={lookup} " if floor is not None else ""
            print(f"\nWINNER[{anchor}]: {ftag}{g}  pooled {base_pooled:.3f} -> {pooled:.3f}")
            print(f"{'cell':<32}{'current':>8}{'winner':>8}")
            for c in sorted(set(base_pc) | set(pc)):
                tag = " *" if c in healthy else ""
                print(f"{c[0]:<22} L{c[1]} t{c[2]}  {base_pc.get(c, float('nan')):>7.2f} "
                      f"{pc.get(c, float('nan')):>7.2f}{tag}")


if __name__ == "__main__":
    main()
