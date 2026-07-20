"""FOL-COMPILE v2 oracle probe: does runtime predicate composition have ANY
headroom over the flat scorer + class rules on SafeLIBERO / LIBERO-Safety?

The proposal under test (user spec): scene graph + VLM critic selects from a
groundable predicate library and COMPOSES rules at runtime -> steering.
This probe evaluates the arm's CEILING: every predicate grounded by an ORACLE
(GT positions; MOVING(x) from the bddl :dynamics block; HAZPROP(x) from the
rated VLM priors; PROTECTED(x) by class; NEAR_PATH from true geometry) and the
rule set composed generously:

  avoid(x) <- MOVING(x)                                 (dynamic intruder)
  avoid(x) <- PROTECTED(x)                              (body-part class)
  avoid(x) <- HAZPROP(x) & NEAR_PATH(x) & ~MENTIONED(x) (property hazard)
  avoid(x) <- NEAR(x, goal_path_2nd_leg) & ~MENTIONED(x) (long-horizon block)

Selected set ranked by the v3 score; metric = top-1/top-2 hit vs the GT
hazard set (SafeLIBERO: designated obstacle; LS: bddl constraint entities).
A real VLM critic can only ground these predicates WORSE than the oracle, so
oracle parity/loss vs flat+classes = airtight NO-GO for accuracy claims.
"""

import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import guard_set as gs  # noqa: E402
import ls_v2_graph_validation as v2  # noqa: E402
import symbolic_identity as si  # noqa: E402
import vlm_slot_bench as vb  # noqa: E402


def moving_oracle(bddl_path):
    try:
        txt = open(bddl_path).read()
        m = re.search(r"\(:dynamics(.*?)\n  \)", txt, re.S) or re.search(r"\(:dynamics(.*)", txt, re.S)
        if not m:
            return set()
        return set(re.findall(r"\(([a-z_0-9]+)\s*\n", m.group(1)))
    except Exception:  # noqa: BLE001
        return set()


def fol_select(task, cands, eef, moving, sigma=0.25, haz_t=0.55, path_t=0.20):
    """Oracle-grounded composed-rule selection; returns ranked avoid list."""
    task_l = task.lower()

    def clean(k):
        return si._clean_object_name(k).lower()

    def mentioned(k):
        toks = [w for w in clean(k).split() if len(w) >= 3 and w != "obstacle"]
        return bool(toks) and (toks[-1] in task_l or all(w in task_l for w in toks))

    goals = [k for k in cands if mentioned(k)]
    spos = np.asarray(eef)[:2]
    segs = [(spos, np.asarray(cands[g])[:2]) for g in goals] or [(spos, np.array([0.0, 0.15]))]
    # 2nd-leg segments (goal_i -> goal_j) for long-horizon rule
    for i, gi in enumerate(goals):
        for gj in goals[i + 1:]:
            segs.append((np.asarray(cands[gi])[:2], np.asarray(cands[gj])[:2]))

    def d_path(k):
        p = np.asarray(cands[k])[:2]
        best = np.inf
        for s, e in segs:
            v = e - s
            l2 = float(v @ v)
            t = 0.0 if l2 < 1e-9 else float(np.clip((p - s) @ v / l2, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(p - (s + t * v))))
        return best

    selected = {}
    for k in cands:
        c = clean(k)
        protected = any(w in c for w in si._PROTECTED_CLASSES)
        is_moving = any(k == m or k.startswith(m) for m in moving)
        hazp = si._hazard_weight(k) >= haz_t
        near = d_path(k) < path_t
        ment = mentioned(k)
        fire = (is_moving or protected
                or (hazp and near and not ment)
                or (near and not ment))
        if fire:
            path = float(np.exp(-d_path(k) ** 2 / (2 * sigma ** 2)))
            w = 1.0 if (protected or is_moving) else si._hazard_weight(k)
            if protected or is_moving:
                path = max(path, 0.6)
            selected[k] = w * path
    return [k for k, _ in sorted(selected.items(), key=lambda kv: -kv[1])]


def main():
    si.load_vlm_priors(str(v2.PRIORS), floor=0.5)

    # ---- SafeLIBERO (19 scenes, GT = designated obstacle) ----
    eps = vb.load_episodes(["safelibero_spatial"], 5)
    fol1 = fol2 = flat1 = flat2 = 0
    for e in eps:
        rank = fol_select(e["task"], e["cands"], e["eef"], set())
        fol1 += int(bool(rank) and rank[0] == e["gt"])
        fol2 += int(e["gt"] in rank[:2])
        flat1 += int(si.scored_obstacle_id(e["task"], e["cands"], e["eef"]) == e["gt"])
        flat2 += int(e["gt"] in gs.guard_set(e["task"], e["cands"], e["eef"], k=2))
    print(json.dumps({"safelibero": {"n": len(eps),
                                     "fol_oracle": [fol1, fol2],
                                     "flat+classes": [flat1, flat2]}}))

    # ---- LIBERO-Safety (constraint-entity GT) ----
    per = {}
    for line in open(v2.SCENES):
        r = json.loads(line)
        if "fail" in r or r["suite"] not in ("obstacle_avoidance", "human_safety"):
            continue
        cands = {k: np.asarray(v) for k, v in r["obj_pos"].items()}
        gt = v2.constraint_entities(r["bddl"], list(cands))
        if not gt:
            continue
        mov = moving_oracle(r["bddl"])
        rank = fol_select(r["language"], cands, np.asarray(r["eef_pos"]), mov)
        f1 = si.scored_obstacle_id(r["language"], cands, np.asarray(r["eef_pos"]))
        f2 = gs.guard_set(r["language"], cands, np.asarray(r["eef_pos"]), k=2)
        st = per.setdefault(r["suite"], {"n": 0, "fol1": 0, "fol2": 0, "flat1": 0, "flat2": 0})
        st["n"] += 1
        st["fol1"] += int(bool(rank) and rank[0] in gt)
        st["fol2"] += int(any(x in gt for x in rank[:2]))
        st["flat1"] += int(f1 in gt)
        st["flat2"] += int(any(x in gt for x in f2))
    print(json.dumps(per, indent=1))


if __name__ == "__main__":
    main()
