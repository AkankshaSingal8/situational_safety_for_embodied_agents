"""Guard-set identification: top-k unmentioned candidates by the v3 score.

Motivation (E5 fail-direction, ledger 2026-07-19 late wave): single-obstacle
guidance makes EVERY identification error fail-unsafe — the true hazard goes
unguarded. The guard set converts argmax identification (3/19 under inverted
priors) into a set with 19/19 recall under every corruption tested (intact /
30%-dropped / fully inverted priors), because path geometry alone keeps the
true obstacle in the top-2; the property prior only orders within the set.

Client integration: send guard_set(...) as `obstacles` (list of per-obstacle
center/radius/scales); server barrier = min over obstacles (same min-composition
as companion points). Pending server support, the [0] element reproduces the
current single-obstacle behavior exactly.
"""

import numpy as np

import symbolic_identity as si


def guard_set(task_description: str, candidate_positions: dict,
              eef_pos, k: int = 2):
    """Top-k unmentioned candidates by hazard x path-proximity x mention score.

    Returns a list of names, best first ([] if everything is task-mentioned).
    guard_set(...)[0:1] == [scored_obstacle_id(...)] by construction.
    """
    task = task_description.lower()

    def mf(key):
        toks = [w for w in si._clean_object_name(key).lower().split()
                if len(w) >= 3 and w != "obstacle"]
        if not toks:
            return 0.0
        f = sum(w in task for w in toks) / len(toks)
        if toks[-1] in task:
            f = 1.0
        return f

    fr = {n: mf(n) for n in candidate_positions}
    # Protected-class override — mirror of scored_obstacle_id (human_safety V2
    # finding): body-part classes are never mention-excluded, path-floored,
    # weight 1.0.
    protected = {n for n in candidate_positions
                 if any(w in si._clean_object_name(n).lower()
                        for w in si._PROTECTED_CLASSES)}
    for n in protected:
        fr[n] = 0.0
    part = [n for n, f in fr.items() if f < 1.0]
    if not part:
        return []
    goals = [n for n, f in fr.items() if f >= 0.5]
    spos = np.asarray(eef_pos)[:2]
    segs = [(spos, np.asarray(candidate_positions[g])[:2]) for g in goals] or \
           [(spos, np.array([0.0, 0.15]))]

    def dp(n):
        p = np.asarray(candidate_positions[n])[:2]
        best = np.inf
        for s, g in segs:
            v = g - s
            l2 = float(v @ v)
            t = 0.0 if l2 < 1e-9 else float(np.clip((p - s) @ v / l2, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(p - (s + t * v))))
        return best

    def score(n):
        path = float(np.exp(-dp(n) ** 2 / (2 * 0.25 ** 2)))
        w = si._hazard_weight(n)
        if n in protected:
            path = max(path, si._PROTECTED_PATH_FLOOR)
            w = 1.0
        return w * path * (1.0 - 0.8 * fr[n])

    return sorted(part, key=score, reverse=True)[:k]
