"""LS ident arms v3 (user-directed): flat scorer vs scene-graph vs hard-FOL
on LIBERO-Safety captured scenes, GT = CheckRobotContact args of the bddl
:constraints block (the suite's own violation predicate — NOT all constraint
entities, which include the carried task object via CheckContact).

Arms (same inputs: language, {name: pos}, eef):
  flat   v3 scored_obstacle_id + guard-set top-2 (VLM priors floor 0.5,
         protected-class rule; MOVING rule inert offline — single snapshot)
  graph  scene_graph.graph_obstacle_id keep_out top-2
  fol    hard boolean rules, v17 lineage:
           HAZARD(x) := PROTECTED(x) | (~MENTIONED(x) & NEAREST_TO_PATH(x))
         rank: protected first (by path dist), then unmentioned by path dist.

Outputs top1/top2 per suite per arm, plus the DISAGREEMENT set (scenes where
top-2 sets differ) — the only scenes where a rollout-level difference between
arms is even possible (identical payload => identical rollout).
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


def hazard_entities(bddl_path, obj_names):
    """Args of CheckRobotContact nodes only — matches the eval client."""
    txt = open(bddl_path).read()
    m = re.search(r"\(:constraints(.*?)\n\)", txt, re.S)
    if not m:
        return set()
    ents = set()
    for arg in re.findall(r"\(\s*CheckRobotContact\s+([A-Za-z0-9_ ]+?)\s*\)",
                          m.group(1), re.I):
        for t in arg.split():
            for o in obj_names:
                if t == o or t == o.rsplit("_", 1)[0]:
                    ents.add(o)
    return ents


def _head_mentioned(name, language):
    base = re.sub(r"(__\d+)?(_\d+)?$", "", name)  # can_of_soda__6_1 -> can_of_soda
    head = base.split("_")[-1].lower()
    return head in language.lower()


def fol_top2(language, cands, eef):
    """Hard-FOL arm: boolean predicates + argmin path distance, no priors."""
    tgt = si.parse_target_heuristic(language, list(cands))
    tpos = cands.get(tgt, eef)

    def d_path(k):
        p, a, b = cands[k], np.asarray(eef), np.asarray(tpos)
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
        return float(np.linalg.norm(p - (a + t * ab)))

    protected = [k for k in cands
                 if any(w in k.lower() for w in si._PROTECTED_CLASSES)]
    obstacles = [k for k in cands
                 if k not in protected and k != tgt
                 and not _head_mentioned(k, language)]
    ranked = (sorted(protected, key=d_path)
              + sorted(obstacles, key=d_path))
    return ranked[:2]


def main():
    si.load_vlm_priors(str(PRIORS), floor=0.5)
    per_suite, disagreements = {}, []
    for line in open(SCENES):
        r = json.loads(line)
        if "fail" in r:
            continue
        cands = {k: np.asarray(v) for k, v in r["obj_pos"].items()}
        gt = hazard_entities(r["bddl"], list(cands))
        if not gt:
            continue
        eef = np.asarray(r["eef_pos"])
        arms = {
            "flat": gs.guard_set(r["language"], cands, eef, k=2),
            "graph": [x[0] for x in sg.hazard_rules(
                sg.build_graph(r["language"], cands, eef))
                if x[1] == "keep_out"][:2],
            "fol": fol_top2(r["language"], cands, eef),
        }
        st = per_suite.setdefault(r["suite"], {"n": 0})
        st["n"] += 1
        for a, top2 in arms.items():
            st[f"{a}1"] = st.get(f"{a}1", 0) + int(bool(top2) and top2[0] in gt)
            st[f"{a}2"] = st.get(f"{a}2", 0) + int(any(x in gt for x in top2))
        if len({tuple(sorted(v)) for v in arms.values()}) > 1:
            disagreements.append({"suite": r["suite"], "task": r["task"],
                                  "gt": sorted(gt), **arms})
    print(json.dumps(per_suite, indent=1))
    print(f"\nDISAGREEMENTS ({len(disagreements)}):")
    for d in disagreements:
        print(json.dumps(d))


if __name__ == "__main__":
    main()
