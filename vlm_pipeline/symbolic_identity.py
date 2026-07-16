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

import re
from typing import Dict, List, Optional

import numpy as np


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


def identify_obstacle(task_description: str, obs, workspace=((-0.5, 0.5), (-0.5, 0.5))) -> Optional[str]:
    """Client entry point: candidates from `*_pos` obs keys (workspace-filtered,
    robot excluded), target via the pickup-phrase heuristic, then symbolic id."""
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
    target = parse_target_heuristic(task_description, list(cands))
    return symbolic_obstacle_id(task_description, cands, target,
                                np.asarray(obs["robot0_eef_pos"]))
