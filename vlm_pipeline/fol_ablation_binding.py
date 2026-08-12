"""Offline binding matrix for the FOL ablation study (task-2-brief.md).

Purpose: GPU rollouts are only worth spending on (ablation, suite) cells
whose guard set actually DIFFERS from the incumbent (frozen, ablation="none")
config on captured scenes -- if the top-2 guard set is identical for every
scene in a cell, a rollout there can only reproduce the incumbent's numbers.
This script computes that binding matrix offline, no GPU/env/policy needed.

Inputs: `results_tables/libero_safety_scenes.jsonl` (one JSON record per
captured LIBERO-Safety scene: `suite`, `task`, `level`, `language`, `obj_pos`
{name: [x,y,z]}, `eef_pos`). Loading pattern follows
`vlm_pipeline/ls_v3_ident_arms.py` (the authoritative precedent for this
file): `obj_pos` is the SOLE candidate view fed to every identity arm --
there is no separate "objects_all" vs "objects" split for this scene schema
(that split exists only in `ident_offline_bench.py`'s DIFFERENT
`safelibero_ident_scenes.jsonl` records). This matches the real eval client
(`run_guided_libero_safety_eval.py`'s `object_positions(obs)`): fol/folprop/
folpropvlm all consume the SAME unfiltered `{name: pos}` view.

Ablation dispatch is Task 1's `symbolic_identity.ablated_obstacle_id`
(already merged) -- this script adds NO new identity logic, only offline
scoring/aggregation over it.

Suite -> id_source routing mirrors the frozen SLURM noGT config
(`slurm/ls_vanilla_tier.slurm`, 2026-08-12 "VANILLA TIER" comment: "same
class-driven routing as the frozen board noGT row (all eng 0.30; oa/oah fol
[+cone60 oah], hs folprop, aff folpropvlm w/ cached VLM tables)") combined
with task-2-brief.md's explicit spec (oa/oah -> fol; hs/aff -> folpropvlm,
also folprop for aff as a secondary informational arm). Where the two
sources disagree on human_safety's SECONDARY arm, the brief (this task's
literal spec) wins; folprop is reported for affordance only, as instructed.
`reasoning_safety` is not named in either source's routing table -- LS ships
five suites but the brief's abbreviations (oa/oah/hs/aff) cover four. Since
reasoning_safety's hazard is PROPERTY/CLASS-gated (unsafe-by-design
instructions), not path-geometric, it is routed like human_safety
(folpropvlm), documented here as an explicit assumption -- flagged in the
report, not hidden in scoring logic.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import symbolic_identity as si  # noqa: E402

ROOT = pathlib.Path(__file__).parents[1]
SCENES = ROOT / "results_tables/libero_safety_scenes.jsonl"
OUT_JSON = ROOT / "results_tables/fol_ablation_binding.json"

GUARD_TOPK = 2  # frozen config (task-2-brief.md)
ABLATIONS = ["geom_only", "no_exempt", "no_class", "no_heat_gate"]

# suite -> list of (id_source, is_secondary) to evaluate. See module
# docstring for the routing-source discussion.
SUITE_ROUTING = {
    "obstacle_avoidance": [("fol", False)],
    "obstacle_avoidance_human": [("fol", False)],
    "human_safety": [("folpropvlm", False)],
    "reasoning_safety": [("folpropvlm", False)],  # assumption, see docstring
    "affordance": [("folpropvlm", False), ("folprop", True)],
}


def mover_variants(candidates):
    """[None] + one {name} per PROTECTED_CLASSES-token candidate name (the
    brief's approximation of the runtime mover case; scenes carry no motion
    info). Note: since every ablation/incumbent function's PROTECTED(x) test
    is `name-token-match OR name in moving`, setting `moving={name}` for a
    name that already matches a PROTECTED_CLASSES token is a structural
    no-op -- consistent with ls_v3_ident_arms.py's "MOVING rule inert
    offline -- single snapshot" note. Implemented literally per spec."""
    variants = [None]
    for name in candidates:
        if any(tok in name.lower() for tok in si._PROTECTED_CLASSES):
            variants.append({name})
    return variants


def eval_cell(id_source, ablation, desc, cands, eef):
    """Returns (binds: bool, detail: dict) for one (id_source, ablation,
    scene). `binds` = True if ANY mover variant's ablated guard set (as a
    SET) differs from the incumbent guard set (moving=None, frozen)."""
    incumbent = si.ablated_obstacle_id(id_source, "none", desc, cands, eef,
                                       moving=None)[:GUARD_TOPK]
    inc_set = set(incumbent)
    binds = False
    variants = []
    for moving in mover_variants(cands):
        ablated = si.ablated_obstacle_id(id_source, ablation, desc, cands,
                                         eef, moving=moving)[:GUARD_TOPK]
        set_diff = set(ablated) != inc_set
        ordered_diff = ablated != incumbent
        binds = binds or set_diff
        variants.append({
            "moving": sorted(moving) if moving else None,
            "ablated_set": ablated,
            "set_diff": set_diff,
            "ordered_diff": ordered_diff,
        })
    return binds, {
        "incumbent_set": incumbent,
        "variants": variants,
    }


def run_matrix(records, routing=SUITE_ROUTING):
    """records: iterable of scene dicts. Returns (cells, skipped_suites)."""
    cells = {}
    skipped = set()
    for r in records:
        if "fail" in r:
            continue
        suite = r["suite"]
        arms = routing.get(suite)
        if arms is None:
            skipped.add(suite)
            continue
        cands = {k: np.asarray(v) for k, v in r["obj_pos"].items()}
        eef = np.asarray(r["eef_pos"])
        desc = r["language"]
        task = r["task"]
        for id_source, secondary in arms:
            for ablation in ABLATIONS:
                key = (suite, id_source, ablation)
                cell = cells.setdefault(key, {
                    "suite": suite, "id_source": id_source,
                    "secondary": secondary, "ablation": ablation,
                    "n_scenes": 0, "n_binding": 0, "details": [],
                })
                binds, detail = eval_cell(id_source, ablation, desc, cands, eef)
                cell["n_scenes"] += 1
                cell["n_binding"] += int(binds)
                cell["details"].append({"task": task, "binds": binds, **detail})
    for key, cell in cells.items():
        cell["binding_frac"] = cell["n_binding"] / max(cell["n_scenes"], 1)
    return cells, skipped


def print_matrix(cells):
    rows = sorted({(c["suite"], c["id_source"]) for c in cells.values()})
    print(f"\n{'suite/id_source':<32}" + "".join(f"{a:>14}" for a in ABLATIONS))
    for suite, id_source in rows:
        line = f"{suite + '/' + id_source:<32}"
        for ab in ABLATIONS:
            c = cells.get((suite, id_source, ab))
            frac = c["binding_frac"] if c else float("nan")
            n = c["n_scenes"] if c else 0
            line += f"{frac:>9.2f}({n:>2})"
        print(line)


def recommended_cells(cells):
    rec = [{"suite": c["suite"], "id_source": c["id_source"],
            "ablation": c["ablation"], "binding_frac": c["binding_frac"],
            "n_scenes": c["n_scenes"], "n_binding": c["n_binding"]}
           for c in cells.values() if c["binding_frac"] > 0]
    rec.sort(key=lambda x: -x["binding_frac"])
    return rec


# --------------------------------------------------------------- selftest

def _selftest():
    """2 inline synthetic records, no file/disk dependency (id_source="fol"
    only -- no VLM tables needed). Asserts one KNOWN binding cell and one
    KNOWN non-binding cell.

    Scene: "pick up the apple and put it on the plate" with candidates
    {apple_1 (mentioned/target), plate_1 (mentioned/destination),
    human_1 (protected-class name), vase_1 (unmentioned, near path)}.

    fol incumbent (ablation="none"): protected=[human_1], obstacles=[vase_1]
    (apple/plate excluded as mentioned) -> top-2 = [human_1, vase_1].

    KNOWN BINDING: ablation="no_class" drops the PROTECTED|MOVING disjunct
    entirely -> returns only the PRED-branch obstacles = [vase_1]. Set
    {vase_1} != {human_1, vase_1} -> binds=True.

    KNOWN NON-BINDING: ablation="no_heat_gate" for id_source="fol" delegates
    BYTE-IDENTICALLY to `ablated_obstacle_id("fol", "none", ...)` per
    `_ablate_no_heat_gate`'s explicit fol/folprop passthrough -- guaranteed
    binds=False by construction, no VLM tables required.
    """
    desc = "pick up the apple and put it on the plate"
    cands = {
        "apple_1": np.array([0.0, 0.15, 0.9]),
        "plate_1": np.array([0.15, 0.15, 0.9]),
        "human_1": np.array([-0.05, 0.05, 0.95]),
        "vase_1": np.array([-0.05, 0.10, 0.9]),
    }
    eef = np.array([-0.21, 0.0, 1.0])

    incumbent = si.ablated_obstacle_id("fol", "none", desc, cands, eef,
                                       moving=None)[:GUARD_TOPK]
    assert set(incumbent) == {"human_1", "vase_1"}, incumbent

    binds_no_class, detail_nc = eval_cell("fol", "no_class", desc, cands, eef)
    assert binds_no_class, detail_nc
    assert set(detail_nc["variants"][0]["ablated_set"]) == {"vase_1"}

    binds_no_heat, detail_nh = eval_cell("fol", "no_heat_gate", desc, cands, eef)
    assert not binds_no_heat, detail_nh
    for v in detail_nh["variants"]:
        assert not v["set_diff"] and not v["ordered_diff"]

    # Exercise run_matrix()/print_matrix()/recommended_cells() end-to-end on
    # the 2 synthetic records (both same scene content, 2 "tasks") without
    # touching results_tables/.
    rec_a = {"suite": "obstacle_avoidance", "task": "synthetic_a", "level": 0,
             "language": desc, "obj_pos": {k: v.tolist() for k, v in cands.items()},
             "eef_pos": eef.tolist()}
    rec_b = dict(rec_a, task="synthetic_b")
    cells, skipped = run_matrix([rec_a, rec_b],
                                routing={"obstacle_avoidance": [("fol", False)]})
    assert not skipped
    nc_cell = cells[("obstacle_avoidance", "fol", "no_class")]
    assert nc_cell["n_scenes"] == 2 and nc_cell["n_binding"] == 2
    assert nc_cell["binding_frac"] == 1.0
    nh_cell = cells[("obstacle_avoidance", "fol", "no_heat_gate")]
    assert nh_cell["n_binding"] == 0 and nh_cell["binding_frac"] == 0.0
    rec = recommended_cells(cells)
    assert all(r["ablation"] != "no_heat_gate" for r in rec)
    assert any(r["ablation"] == "no_class" for r in rec)
    print_matrix(cells)

    print("\nSELFTEST OK: known binding (no_class) and known non-binding "
          "(no_heat_gate, byte-identical delegation) both verified.")


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=str, default=str(SCENES))
    ap.add_argument("--out", type=str, default=str(OUT_JSON))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    records = [json.loads(line) for line in open(args.scenes) if line.strip()]
    cells, skipped = run_matrix(records)
    if skipped:
        print(f"NOTE: skipped suites with no routing entry: {sorted(skipped)}")

    print_matrix(cells)
    rec = recommended_cells(cells)
    print(f"\nRECOMMENDED ROLLOUT CELLS ({len(rec)}, binding_frac > 0, "
          f"desc order):")
    for r in rec:
        print(f"  {r['suite']:<24} {r['id_source']:<11} {r['ablation']:<13} "
              f"binding_frac={r['binding_frac']:.2f} "
              f"({r['n_binding']}/{r['n_scenes']})")

    out_obj = {
        "cells": [dict(c, key=f"{c['suite']}|{c['id_source']}|{c['ablation']}")
                  for c in cells.values()],
        "recommended_rollout_cells": rec,
        "skipped_suites": sorted(skipped),
        "guard_topk": GUARD_TOPK,
    }
    with open(args.out, "w") as f:
        json.dump(out_obj, f, indent=1)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
