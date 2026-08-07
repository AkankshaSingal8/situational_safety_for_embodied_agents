"""Symbolic hazard identification (Tier-SemID identity leg) — vendored from
`.worktrees/fol-safety-filter/fol_safety_filter/vlm_grounder.py` (v17, validated
21/21 vs GT on saved safelibero_spatial episodes 2026-07-03; whole-scene VLM
exclusion prompting scored 0-10% on the same data).

OBSTACLE(x) := NOT MENTIONED_IN_TASK(x) AND NOT SUPPORT(x) AND NEAREST_TO_PATH(x)

No `_obstacle` naming convention is consulted: `_clean_object_name` strips the
suffix BEFORE any matching, so identity is decided purely from (a) which scene
object names appear in the instruction and (b) geometry relative to the
eef→target reach path. Object names+positions come from the sim (Tier-SemID);
a detector/VLM supplies them in the full Tier-Percep stack.
"""

import os
import re
from typing import Dict, List, Optional

import numpy as np

# E5 fail-direction stress test (fail-safe figure): corrupt the METHOD's
# identity on purpose and measure which axis (CAR vs TSR) absorbs the error.
# Activated ONLY via env var — default behavior is bit-identical.
_E5_CORRUPT = os.environ.get("E5_CORRUPT", "")


def _clean_object_name(obs_key: str) -> str:
    name = re.sub(r"_obstacle", "", obs_key)
    name = re.sub(r"_\d+$", "", name)
    return name.replace("_", " ").strip()


def parse_target_heuristic(description: str, candidates: List[str]) -> Optional[str]:
    if not description:
        return None
    desc = description.lower()
    pickup_patterns = [
        r"pick up (?:the )?(.+?)(?:\s+(?:between|on|from|in|and|that|which)|$)",
        r"grab (?:the )?(.+?)(?:\s+(?:between|on|from|in|and|that|which)|$)",
        r"move (?:the )?(.+?) (?:to|onto|into)",
        r"put (?:the )?(.+?) (?:on|in|into|onto)",
        r"place (?:the )?(.+?) (?:on|in|into|onto)",
    ]
    for pattern in pickup_patterns:
        m = re.search(pattern, desc)
        if m:
            phrase = m.group(1).strip().rstrip(".,;")
            best_match, best_score = None, 0
            for cand in candidates:
                cleaned = _clean_object_name(cand).lower()
                words = cleaned.split()
                for start in range(len(words)):
                    suffix = " ".join(words[start:])
                    if suffix and suffix in phrase and len(suffix) > best_score:
                        best_match, best_score = cand, len(suffix)
            if best_match:
                return best_match
    return None


def symbolic_obstacle_id(
    task_description: str,
    candidate_positions: Dict[str, "np.ndarray"],
    target_obs_key: Optional[str] = None,
    eef_pos: Optional["np.ndarray"] = None,
) -> Optional[str]:
    task = task_description.lower()

    def mentioned(key: str) -> bool:
        clean = _clean_object_name(key).lower()
        words = [w for w in clean.split() if len(w) > 3]
        return any(w in task for w in words)

    unmentioned = [k for k in candidate_positions if not mentioned(k)]
    if not unmentioned:
        return None

    # ON_TOP_OF(a, b): b is a support pedestal, not the hazard itself.
    supports = set()
    for a in unmentioned:
        pa = np.asarray(candidate_positions[a])
        for b in unmentioned:
            if a == b:
                continue
            pb = np.asarray(candidate_positions[b])
            if (float(np.linalg.norm(pa[:2] - pb[:2])) < 0.10
                    and float(pa[2]) > float(pb[2]) + 0.04):
                supports.add(b)
    unmentioned = [k for k in unmentioned if k not in supports] or unmentioned

    if len(unmentioned) == 1:
        return unmentioned[0]

    if target_obs_key and target_obs_key in candidate_positions:
        tpos = np.asarray(candidate_positions[target_obs_key])[:2]
    else:
        tpos = np.array([0.0, 0.15])

    spos = np.asarray(eef_pos)[:2] if eef_pos is not None else np.array([-0.21, 0.0])
    seg = tpos - spos
    seg_len2 = float(seg @ seg)

    def dist_to_path(k: str) -> float:
        p = np.asarray(candidate_positions[k])[:2]
        if seg_len2 < 1e-9:
            return float(np.linalg.norm(p - tpos))
        t = float(np.clip((p - spos) @ seg / seg_len2, 0.0, 1.0))
        return float(np.linalg.norm(p - (spos + t * seg)))

    return min(unmentioned, key=dist_to_path)


# Hazard-class property prior (C2's property head, name-token based; the VLM
# supplies these attributes in the full Tier-Percep stack). Higher = more
# hazard-like; groceries/servingware are benign manipulanda.
HAZARD_PRIOR = {
    "bottle": 0.9, "wine": 0.9, "glass": 0.9,
    "pot": 0.9, "moka": 0.9, "kettle": 0.9, "pan": 0.85,
    "mug": 0.7, "cup": 0.7,
    "milk": 0.7, "storage": 0.6, "box": 0.5, "book": 0.5,
    "plate": 0.1, "bowl": 0.15, "basket": 0.1, "ramekin": 0.15, "tray": 0.1,
    "pudding": 0.2, "cheese": 0.2, "butter": 0.2, "sauce": 0.2, "soup": 0.2,
    "ketchup": 0.2, "juice": 0.2, "cookies": 0.2, "bbq": 0.2, "cream": 0.2,
}
_HAZARD_DEFAULT = 0.4


# VLM-sourced property priors (generalization arm G1): per-clean-name hazard
# ratings generated OFFLINE by a VLM (vlm_slot_bench rate_name, anchored
# prompt) + a protection floor — semantics may RAISE concern above the floor
# but never delete protection from an unmentioned near-path object (C1
# fail-safe principle; floor=0.5 was the best offline config, 17/19 vs
# table 19/19 with full open-vocab generality the table lacks).
_VLM_PRIORS: dict | None = None
_PRIOR_FLOOR = 0.5


def load_vlm_priors(path: str, floor: float = 0.5) -> None:
    global _VLM_PRIORS, _PRIOR_FLOOR
    import json
    _VLM_PRIORS = {k.lower(): float(v) for k, v in json.load(open(path)).items()}
    _PRIOR_FLOOR = floor


def _hazard_weight(name: str) -> float:
    clean = _clean_object_name(name).lower()
    if _VLM_PRIORS is not None:
        base = clean.replace(" obstacle", "")
        rated = _VLM_PRIORS.get(base)
        if rated is None:
            # Composite-name fallback (ident bench 2026-07-31): exact lookup
            # missed names like "moka pot small" (only "moka pot" rated),
            # collapsing the true hazard to the unseen-name default while
            # inflated grocery ratings won. Substring match mirrors the hand
            # table's token-max behavior. No GT hazard knowledge consulted.
            hits = [v for k, v in _VLM_PRIORS.items() if k in base or base in k]
            rated = max(hits) if hits else _HAZARD_DEFAULT
        return max(rated, _PRIOR_FLOOR)
    toks = clean.split()
    ws = [HAZARD_PRIOR[t] for t in toks if t in HAZARD_PRIOR]
    return max(ws) if ws else _HAZARD_DEFAULT


# Semantic hazard CLASSES guarded unconditionally: never mention-excluded,
# path term floored (dynamic intruders), hazard weight 1.0. Extend via
# rated properties, not names — these are body-part classes, not objects.
_PROTECTED_CLASSES = ("hand", "human", "person", "arm", "finger", "face")
_PROTECTED_PATH_FLOOR = 0.5

# Prior-entry calibration (offline ident bench 2026-07-31, 320 captured
# SafeLIBERO scenes, results_tables/safelibero_ident_scenes.jsonl): the VLM
# arm's pooled top-1 went 0.637 -> 0.787 with prior^1.5 + a low-prior veto,
# with zero regression on any healthy cell (>=0.85) of either prior source.
# ALPHA sharpens hazard/benign contrast against the path term; TAU bars
# benign-class objects (floored plates/boxes) from the near-path argmax
# unless PROTECTED/MOVING, with a fail-safe fallback if it empties the pool.
_PRIOR_ALPHA = 1.5
_VETO_TAU = 0.6


def scored_obstacle_id(
    task_description: str,
    candidate_positions: Dict[str, "np.ndarray"],
    eef_pos: Optional["np.ndarray"] = None,
    sigma: float = 0.25,
    moving: Optional[set] = None,
) -> Optional[str]:
    """v3 identity for distractor-rich scenes: soft scoring instead of hard
    mention exclusion.

    score(x) = hazard_prior(x) · exp(−d_path(x)²/2σ²) · (1 − 0.8·mention_frac(x))

    where mention_frac = fraction of the name's content tokens present in the
    instruction (full mention ⇒ hard-excluded), and d_path = min xy distance
    to any reach segment eef→mentioned-object (multi-goal aware — fixes Long
    tasks where the obstacle blocks the SECOND goal's path).
    """
    task = task_description.lower()

    def mention_frac(key: str) -> float:
        toks = [w for w in _clean_object_name(key).lower().split()
                if len(w) >= 3 and w != "obstacle"]
        if not toks:
            return 0.0
        frac = sum(w in task for w in toks) / len(toks)
        # Head-noun rule: multi-word visual descriptors ("glazed rim porcelain
        # ramekin") dilute token-fraction matching — if the name's head noun
        # (last content token, English NP convention) appears in the
        # instruction, the object IS referenced and must be goal-excluded.
        if toks[-1] in task:
            frac = 1.0
        return frac

    fracs = {k: mention_frac(k) for k in candidate_positions}
    # Protected-class override (LIBERO-Safety human_safety V2 finding,
    # 2026-07-20): a human body part is hazardous by CLASS, not by geometry —
    # it must never be mention-excluded ("place it in my hand" names the hand
    # as a destination, not a license to touch it) and never path-gated to
    # zero (it intrudes DYNAMICALLY; episode-start distance is meaningless).
    # Offline: flat scorer 0/15 -> 14/15 top-1, 15/15 top-2 with this rule.
    protected = {k for k in candidate_positions
                 if any(w in _clean_object_name(k).lower()
                        for w in _PROTECTED_CLASSES)}
    # MOVING class rule (FOL probe survivor #2, ledger 2026-07-21): a dynamic
    # intruder is hazardous by CLASS — grounded at runtime by inter-frame
    # displacement (the caller observes which objects moved during the settle
    # window), never by name. Same override as protected: never
    # mention-excluded, path term floored (its episode-start distance is
    # meaningless), weight 1.0.
    if moving:
        protected = protected | {k for k in candidate_positions if k in moving}
    for k in protected:
        fracs[k] = 0.0
    partial = {k for k, f in fracs.items() if f < 1.0}
    if not partial:
        return None
    goals = [k for k, f in fracs.items() if f >= 0.5]

    spos = np.asarray(eef_pos)[:2] if eef_pos is not None else np.array([-0.21, 0.0])
    segs = []
    for g in goals:
        segs.append((spos, np.asarray(candidate_positions[g])[:2]))
    if not segs:
        segs.append((spos, np.array([0.0, 0.15])))

    def d_path(k: str) -> float:
        p = np.asarray(candidate_positions[k])[:2]
        best = np.inf
        for s, e in segs:
            seg = e - s
            l2 = float(seg @ seg)
            t = 0.0 if l2 < 1e-9 else float(np.clip((p - s) @ seg / l2, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(p - (s + t * seg))))
        return best

    def score(k: str) -> float:
        path = float(np.exp(-d_path(k) ** 2 / (2 * sigma ** 2)))
        w = _hazard_weight(k)
        if k in protected:
            path = max(path, _PROTECTED_PATH_FLOOR)
            w = 1.0
        return w ** _PRIOR_ALPHA * path * (1.0 - 0.8 * fracs[k])

    eligible = [k for k in partial
                if k in protected or _hazard_weight(k) >= _VETO_TAU]
    ranking = sorted(eligible or partial, key=score, reverse=True)
    if _E5_CORRUPT in ("identity", "both") and len(ranking) > 1:
        return ranking[1]  # deliberate second-best pick
    return ranking[0]


def fol_obstacle_id(task_description: str, candidates, eef_pos,
                    moving: Optional[set] = None):
    """Hard-FOL ranking (v17 lineage), no VLM priors:
        HAZARD(x) := PROTECTED(x) | MOVING(x) | (~MENTIONED(x) & NEAR_PATH(x))
    Returns candidates ranked protected/moving first (by path distance), then
    unmentioned by path distance. LS offline validation: 12/15 top-1 on
    obstacle_avoidance vs 9/15 for the prior-weighted scorer; ties 12/12 on
    human_safety (ls_v3_ident_arms.py)."""
    import re as _re
    tgt = parse_target_heuristic(task_description, list(candidates))
    tpos = candidates.get(tgt, eef_pos)

    def d_path(k):
        p, a, b = np.asarray(candidates[k]), np.asarray(eef_pos), np.asarray(tpos)
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
        return float(np.linalg.norm(p - (a + t * ab)))

    def head_mentioned(name):
        base = _re.sub(r"(__\d+)?(_\d+)?$", "", name)
        return base.split("_")[-1].lower() in task_description.lower()

    protected = [k for k in candidates
                 if any(w in k.lower() for w in _PROTECTED_CLASSES)
                 or (moving and k in moving)]
    obstacles = [k for k in candidates
                 if k not in protected and k != tgt and not head_mentioned(k)]
    return sorted(protected, key=d_path) + sorted(obstacles, key=d_path)


# Affordance-suite property predicates (name-token grounded; the VLM property
# head supplies these attributes in the full Tier-Percep stack).
_PROP_HOT = ("stove", "moka", "kettle", "iron", "candle", "lighter", "burner")
_PROP_SHARP = ("knife", "scissors", "blade", "shears", "saw")
_PROP_FRAGILE = ("porcelain", "glass", "vase", "egg", "ceramic", "wine")


def folprop_obstacle_id(task_description: str, candidates, eef_pos,
                        moving: Optional[set] = None):
    """Property-gated FOL ranking for affordance-class tasks:
        HAZARD(x) := PROTECTED(x) | MOVING(x)
                     | (PROP(x) & ~MENTIONED(x) & ~TARGET(x))
    where PROP = HOT | SHARP | FRAGILE (name-token grounded). Unlike
    fol_obstacle_id there is NO fallback to unmentioned near-path objects:
    if nothing is property-positive the returned ranking is EMPTY (correct
    no-op on affordance scenes whose hazard is part-level or absent)."""
    import re as _re
    tgt = parse_target_heuristic(task_description, list(candidates))
    tpos = candidates.get(tgt, eef_pos)

    def d_path(k):
        p, a, b = np.asarray(candidates[k]), np.asarray(eef_pos), np.asarray(tpos)
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
        return float(np.linalg.norm(p - (a + t * ab)))

    def head_mentioned(name):
        base = _re.sub(r"(__\d+)?(_\d+)?$", "", name)
        return base.split("_")[-1].lower() in task_description.lower()

    prop_tokens = _PROP_HOT + _PROP_SHARP + _PROP_FRAGILE
    protected = [k for k in candidates
                 if any(w in k.lower() for w in _PROTECTED_CLASSES)
                 or (moving and k in moving)]
    obstacles = [k for k in candidates
                 if k not in protected and k != tgt and not head_mentioned(k)
                 and any(tok in k.lower() for tok in prop_tokens)]
    return sorted(protected, key=d_path) + sorted(obstacles, key=d_path)


# VLM-grounded property tables (folpropvlm arm): per-base-class boolean
# predicates generated OFFLINE by the VLM property head (N=5 majority-voted,
# results_tables/ls_vlm_{properties,heat_source}_n5.json). Lazy-loaded and
# cached at module level; unit tests inject tables via kwargs instead.
_VLM_PROP_TABLE: Optional[dict] = None
_VLM_HEAT_TABLE: Optional[dict] = None

# HOT is state-dependent: a hot-CLASS object is only an effective hazard when
# a heat source is nearby to have heated it (xy-agnostic 3D distance gate).
_HEAT_PROXIMITY_M = 0.15


def _vlm_base_class(name: str) -> str:
    """Base-class normalization matching the table-build convention."""
    base = re.sub(r"(_\d+)+$", "", name).replace("_", " ").strip()
    base = re.sub(r"\s+\d+(\s|$)", " ", base).strip()
    return base.lower()


def _load_vlm_prop_tables():
    global _VLM_PROP_TABLE, _VLM_HEAT_TABLE
    if _VLM_PROP_TABLE is None or _VLM_HEAT_TABLE is None:
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent / "results_tables"
        with open(root / "ls_vlm_properties_n5.json") as f:
            _VLM_PROP_TABLE = json.load(f)
        with open(root / "ls_vlm_heat_source_n5.json") as f:
            _VLM_HEAT_TABLE = json.load(f)
    return _VLM_PROP_TABLE, _VLM_HEAT_TABLE


def folpropvlm_obstacle_id(task_description: str, candidates, eef_pos,
                           moving: Optional[set] = None,
                           properties: Optional[dict] = None,
                           heat_sources: Optional[dict] = None):
    """VLM-grounded folprop: identical FOL structure to folprop_obstacle_id
        HAZARD(x) := PROTECTED(x) | MOVING(x)
                     | (PROP(x) & ~MENTIONED(x) & ~TARGET(x))
    but PROP(x) is the state-aware VLM rule instead of name tokens:
        prop_effective(x) = SHARP(x) | FRAGILE(x) | HEAT_SOURCE(x)
                            | (HOT(x) & exists s: HEAT_SOURCE(s)
                               & ||pos(x)-pos(s)|| < 0.15)
    Predicates come from the cached N=5 majority-voted VLM tables (or the
    `properties`/`heat_sources` kwargs in tests). A base class absent from a
    table has all properties False. No near-path fallback: may return []."""
    import re as _re
    if properties is None or heat_sources is None:
        props_tab, heat_tab = _load_vlm_prop_tables()
        if properties is None:
            properties = props_tab
        if heat_sources is None:
            heat_sources = heat_tab

    tgt = parse_target_heuristic(task_description, list(candidates))
    tpos = candidates.get(tgt, eef_pos)

    def d_path(k):
        p, a, b = np.asarray(candidates[k]), np.asarray(eef_pos), np.asarray(tpos)
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
        return float(np.linalg.norm(p - (a + t * ab)))

    def head_mentioned(name):
        base = _re.sub(r"(__\d+)?(_\d+)?$", "", name)
        return base.split("_")[-1].lower() in task_description.lower()

    def prop(k, key):
        return bool(properties.get(_vlm_base_class(k), {}).get(key, False))

    def is_heat_source(k):
        return bool(heat_sources.get(_vlm_base_class(k), False))

    heat_pos = [np.asarray(candidates[s]) for s in candidates
                if is_heat_source(s)]

    def prop_effective(k):
        if prop(k, "sharp") or prop(k, "fragile") or is_heat_source(k):
            return True
        if prop(k, "hot"):
            p = np.asarray(candidates[k])
            return any(float(np.linalg.norm(p - hp)) < _HEAT_PROXIMITY_M
                       for hp in heat_pos)
        return False

    protected = [k for k in candidates
                 if any(w in k.lower() for w in _PROTECTED_CLASSES)
                 or (moving and k in moving)]
    obstacles = [k for k in candidates
                 if k not in protected and k != tgt and not head_mentioned(k)
                 and prop_effective(k)]
    return sorted(protected, key=d_path) + sorted(obstacles, key=d_path)


def identify_obstacle(task_description: str, obs, workspace=((-0.5, 0.5), (-0.5, 0.5)),
                      moving: Optional[set] = None) -> Optional[str]:
    """Client entry point: candidates from `*_pos` obs keys (workspace-filtered,
    robot excluded), target via the pickup-phrase heuristic, then symbolic id.
    `moving`: names observed displacing during the settle window (MOVING class)."""
    cands = {}
    for k in obs:
        if not k.endswith("_pos") or k.startswith("robot0"):
            continue
        # relative-pose observables ('X_to_robot0_eef_pos') are NOT objects
        if "_to_" in k:
            continue
        p = np.asarray(obs[k])
        if p.shape != (3,):
            continue
        if (workspace[0][0] < p[0] < workspace[0][1]
                and workspace[1][0] < p[1] < workspace[1][1] and p[2] > 0):
            cands[k[:-4]] = p
    if not cands:
        return None
    return scored_obstacle_id(task_description, cands,
                              np.asarray(obs["robot0_eef_pos"]), moving=moving)
