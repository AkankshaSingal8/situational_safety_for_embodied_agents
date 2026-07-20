"""Scene-graph V2 (pre-registered): flat scorer vs graph rules on LIBERO-Safety.

GT hazard set per task = object arguments of the bddl :constraints block
(bddl_utils.robosuite_parse_problem — the suite's own cost definition), matched
against the captured scene's object inventory. Scenes from
results_tables/libero_safety_scenes.jsonl (ls_v2_capture job).

Arms (same inputs: task language, {name: pos}, eef):
  flat   v3 scored_obstacle_id + guard-set top-2 (VLM priors + floor 0.5;
         unseen names -> default weight)
  graph  scene_graph.graph_obstacle_id + top-2 keep_out rules

Metric: top-1 / top-2 hit against the GT hazard SET, per suite. Decision rule
(pre-registered in the design doc): graph earns integration only with a clear
F1/recall win on the relational suites; parity or loss = NO-GO (as SafeLIBERO
V1 already showed for object-hazard scenes).
"""

import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import guard_set as gs  # noqa: E402
import scene_graph as sg  # noqa: E402
import symbolic_identity as si  # noqa: E402

ROOT = pathlib.Path(__file__).parents[1]
SCENES = ROOT / "results_tables/libero_safety_scenes.jsonl"
PRIORS = ROOT / "results_tables/vlm_hazard_priors.json"
LS = ROOT.parents[1] / "LIBERO-Safety"


def constraint_entities(bddl_path, obj_names):
    """Object args of :constraints predicates, matched to scene objects."""
    txt = open(bddl_path).read()
    m = re.search(r"\(:constraints(.*?)\n\)", txt, re.S)
    if not m:
        return set()
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", m.group(1))
    drop = {"And", "Or", "Not", "constraints"}
    ents = set()
    for t in toks:
        if t in drop or t.lower().startswith("check"):
            continue
        for o in obj_names:
            if t == o or t == o.rsplit("_", 1)[0]:
                ents.add(o)
    return ents


def top2_graph(task, cands, eef):
    g = sg.build_graph(task, cands, eef)
    keep = [r for r in sg.hazard_rules(g) if r[1] == "keep_out"]
    return [r[0] for r in keep[:2]]


def main():
    si.load_vlm_priors(str(PRIORS), floor=0.5)
    per_suite = {}
    rows = []
    for line in open(SCENES):
        r = json.loads(line)
        if "fail" in r:
            continue
        cands = {k: np.asarray(v) for k, v in r["obj_pos"].items()}
        gt = constraint_entities(r["bddl"], list(cands))
        if not gt:
            rows.append({"suite": r["suite"], "task": r["task"], "skip": "no constraint entities"})
            continue
        eef = np.asarray(r["eef_pos"])
        f1 = si.scored_obstacle_id(r["language"], cands, eef)
        f2 = gs.guard_set(r["language"], cands, eef, k=2)
        g2 = top2_graph(r["language"], cands, eef)
        g1 = g2[0] if g2 else None
        st = per_suite.setdefault(r["suite"], {"n": 0, "flat1": 0, "flat2": 0, "graph1": 0, "graph2": 0})
        st["n"] += 1
        st["flat1"] += int(f1 in gt)
        st["flat2"] += int(any(x in gt for x in f2))
        st["graph1"] += int(g1 in gt)
        st["graph2"] += int(any(x in gt for x in g2))
        rows.append({"suite": r["suite"], "task": r["task"], "gt": sorted(gt),
                     "flat": [f1, f2], "graph": g2})
    for r in rows:
        print(json.dumps(r))
    print(json.dumps(per_suite, indent=1))


if __name__ == "__main__":
    main()
