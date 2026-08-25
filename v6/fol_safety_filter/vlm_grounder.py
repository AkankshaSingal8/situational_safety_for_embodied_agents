"""
FOL Safety Filter v6 — 4-Phase CoT VLM Grounder

Chain:
  P1: Scene description   → free text (warm-up reasoning, all views)
  P2: Role identification → JSON {task_target, goal_location, obstacles[]}
                           N=3 majority vote per field
  P3: Property attribution → JSON {properties: {obj: [props...]}}
                             N=3 majority vote per property per object
  P4: Rule composition    → JSON {rules: [...], composed_predicates: [...]}
                            N=1 (synthesis, deterministic given P1-P3 context)

Falls back to v3 heuristic (obstacle substring scan) if VLM unavailable or fails.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .prompts import (
    prompt_scene_description,
    prompt_role_identification,
    prompt_property_attribution,
    prompt_rule_composition,
)
from .vlm_client import VLMClient

logger = logging.getLogger(__name__)

_VALID_PROPERTIES = {"IS_FRAGILE", "IS_HOT", "IS_SPILLABLE", "IS_SHARP", "IS_HEAVY"}

N_VOTES_ROLE = 3
N_VOTES_PROPS = 3


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[Dict]:
    """Extract first valid JSON object from model output (handles extra text)."""
    if not text:
        return None
    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first {...} block
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Name-matching helpers
# ---------------------------------------------------------------------------

def _build_cleaned_map(candidates: List[str]) -> Dict[str, str]:
    """Map cleaned display name → obs_key."""
    cleaned_map: Dict[str, str] = {}
    for name in candidates:
        cleaned = name.replace("_", " ").rstrip(" 0123456789").strip()
        cleaned = cleaned.replace(" obstacle", "").strip()
        if cleaned:
            cleaned_map[cleaned.lower()] = name
    return cleaned_map


def _match_name(
    text: Optional[str],
    cleaned_map: Dict[str, str],
) -> Optional[str]:
    """Match a VLM-returned name string to an obs_key using exact then fuzzy."""
    if not text or str(text).lower() in ("null", "none", "", "unknown"):
        return None
    t = str(text).lower().strip().strip('"\'')
    if t in cleaned_map:
        return cleaned_map[t]
    # Substring match: find the cleaned key that best fits
    best_key = None
    best_score = 0
    for cleaned, obs_key in cleaned_map.items():
        if cleaned in t or t in cleaned:
            score = len(cleaned)
            if score > best_score:
                best_key = obs_key
                best_score = score
    return best_key


# ---------------------------------------------------------------------------
# Majority-vote helpers
# ---------------------------------------------------------------------------

def _majority_string(responses: List[Optional[str]]) -> Optional[str]:
    """Return the most common non-None value, or None."""
    valid = [r for r in responses if r is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def _majority_list(responses: List[List[str]]) -> List[str]:
    """Return items that appear in majority of response lists."""
    if not responses:
        return []
    n = len(responses)
    counter: Counter = Counter()
    for lst in responses:
        counter.update(set(lst))  # count each item once per response
    return [item for item, cnt in counter.items() if cnt > n // 2]


def _majority_props(
    responses: List[Dict[str, List[str]]],
    obstacles: List[str],
) -> Dict[str, List[str]]:
    """Majority vote per property per obstacle across N responses."""
    result: Dict[str, List[str]] = {obs: [] for obs in obstacles}
    n = len(responses)
    for obs in obstacles:
        prop_counts: Counter = Counter()
        for resp in responses:
            props = resp.get(obs, [])
            prop_counts.update(set(p for p in props if p in _VALID_PROPERTIES))
        result[obs] = [p for p, cnt in prop_counts.items() if cnt > n // 2]
    return result


# ---------------------------------------------------------------------------
# Main grounder class
# ---------------------------------------------------------------------------

class VLMObstacleGrounder:
    """
    v6: 4-phase CoT VLM grounder with majority voting.

    Returns predicate_dict compatible with v5/v4 rule_composer and filter.
    """

    def __init__(self, use_vlm: bool = True):
        self.use_vlm = use_vlm
        self._client: Optional[VLMClient] = None

    def _get_client(self) -> VLMClient:
        if self._client is None:
            self._client = VLMClient()
        return self._client

    def detect(
        self,
        rgb_images: Any,
        task_description: str,
        candidate_obs_keys: List[str],
    ) -> Dict:
        if isinstance(rgb_images, np.ndarray):
            rgb_images = [rgb_images]
        else:
            rgb_images = [img for img in (rgb_images or []) if img is not None]

        result = _empty_result()
        cleaned_map = _build_cleaned_map(candidate_obs_keys)
        cleaned_candidates = list(cleaned_map.keys())

        # Parse target from task description as strong prior
        parsed_target_key = _parse_target_heuristic(task_description, candidate_obs_keys)
        if parsed_target_key:
            logger.info(f"[VLMGrounder-v6] Heuristic target parse: {parsed_target_key}")

        client = self._get_client()
        if self.use_vlm and client.backend_name != "mock":
            try:
                vlm_result = self._cot_detect(
                    client, rgb_images, task_description,
                    cleaned_candidates, cleaned_map, parsed_target_key,
                )
                if vlm_result.get("obstacle_obs_keys") is not None:
                    result.update(vlm_result)
                    result["method"] = f"vlm-{client.backend_name}"
                    if vlm_result["obstacle_obs_keys"]:
                        first = vlm_result["obstacle_obs_keys"][0]
                        result["obstacle_obs_key"] = first
                        result["physical_properties"] = vlm_result["obstacle_properties"].get(first, [])
                    logger.info(
                        f"[VLMGrounder-v6] CoT complete | "
                        f"target={result.get('task_target_obs_key')} "
                        f"goal={result.get('goal_location_obs_key')} "
                        f"obstacles={result.get('obstacle_obs_keys')} "
                        f"backend={client.backend_name}"
                    )
                    return result
            except Exception as e:
                logger.warning(f"[VLMGrounder-v6] CoT failed ({e}), falling back to heuristic")

        # Heuristic fallback (v3 behavior)
        return _heuristic_result(candidate_obs_keys, parsed_target_key)

    # ------------------------------------------------------------------ #
    # 4-phase CoT chain
    # ------------------------------------------------------------------ #

    def _cot_detect(
        self,
        client: VLMClient,
        images: List[np.ndarray],
        task_description: str,
        cleaned_candidates: List[str],
        cleaned_map: Dict[str, str],
        parsed_target_key: Optional[str],
    ) -> Dict:
        parsed_target_cleaned = None
        if parsed_target_key:
            parsed_target_cleaned = parsed_target_key.replace("_", " ").rstrip(" 0123456789")
            parsed_target_cleaned = parsed_target_cleaned.replace(" obstacle", "").strip().lower()

        # --- Phase 1: Scene description ---
        p1_prompt = prompt_scene_description(task_description, cleaned_candidates)
        scene_desc = client.infer(p1_prompt, images, max_tokens=400)
        logger.debug(f"[VLMGrounder-v6] P1 scene desc: {scene_desc[:200]}")

        # --- Phase 2: Role identification (N=3 votes) ---
        target_votes: List[Optional[str]] = []
        goal_votes: List[Optional[str]] = []
        obstacle_votes: List[List[str]] = []

        for vote_i in range(N_VOTES_ROLE):
            p2_prompt = prompt_role_identification(task_description, cleaned_candidates, scene_desc)
            raw = client.infer(p2_prompt, images, max_tokens=256)
            parsed = _extract_json(raw)
            if not parsed:
                logger.debug(f"[VLMGrounder-v6] P2 vote {vote_i}: no JSON in: {raw[:100]}")
                target_votes.append(None)
                goal_votes.append(None)
                obstacle_votes.append([])
                continue

            raw_target = parsed.get("task_target")
            raw_goal = parsed.get("goal_location")
            raw_obstacles = parsed.get("obstacles", []) or []

            target_key = _match_name(raw_target, cleaned_map)
            goal_key = _match_name(raw_goal, cleaned_map)
            obs_keys = [k for n in raw_obstacles if (k := _match_name(n, cleaned_map)) is not None]

            # Safety: never let VLM mark parsed_target as an obstacle
            if parsed_target_cleaned:
                obs_keys = [o for o in obs_keys
                            if parsed_target_cleaned not in o.replace("_", " ").lower()]

            target_votes.append(target_key)
            goal_votes.append(goal_key)
            obstacle_votes.append(obs_keys)
            logger.debug(
                f"[VLMGrounder-v6] P2 vote {vote_i}: "
                f"target={target_key} goal={goal_key} obs={obs_keys}"
            )

        # Majority vote on roles
        task_target_key = (
            _majority_string(target_votes)
            or parsed_target_key  # fall back to heuristic parse if VLM disagrees
        )
        goal_location_key = _majority_string(goal_votes)
        obstacle_keys = _majority_list(obstacle_votes)

        # Remove target and goal from obstacles (safety)
        obstacle_keys = [
            o for o in obstacle_keys
            if o != task_target_key and o != goal_location_key
        ]

        logger.info(
            f"[VLMGrounder-v6] P2 result: target={task_target_key} "
            f"goal={goal_location_key} obstacles={obstacle_keys}"
        )

        # If no VLM obstacles, try heuristic scan
        if not obstacle_keys:
            for name in list(cleaned_map.values()):
                if "obstacle" in name.lower() and name != task_target_key:
                    obstacle_keys = [name]
                    break

        # --- Phase 3: Property attribution (N=3 votes) ---
        obstacle_properties: Dict[str, List[str]] = {o: [] for o in obstacle_keys}
        if obstacle_keys:
            cleaned_obstacles = []
            for o in obstacle_keys:
                for cleaned, key in cleaned_map.items():
                    if key == o:
                        cleaned_obstacles.append(cleaned)
                        break

            prop_responses: List[Dict[str, List[str]]] = []
            for vote_i in range(N_VOTES_PROPS):
                p3_prompt = prompt_property_attribution(cleaned_obstacles, scene_desc)
                raw = client.infer(p3_prompt, images, max_tokens=256)
                parsed = _extract_json(raw)
                if not parsed:
                    prop_responses.append({})
                    continue
                raw_props = parsed.get("properties", {}) or {}
                # Map cleaned names back to obs keys
                vote_props: Dict[str, List[str]] = {}
                for raw_name, props in raw_props.items():
                    obs_key = _match_name(raw_name, cleaned_map)
                    if obs_key and obs_key in obstacle_keys:
                        vote_props[obs_key] = [p for p in (props or []) if p in _VALID_PROPERTIES]
                prop_responses.append(vote_props)
                logger.debug(f"[VLMGrounder-v6] P3 vote {vote_i}: {vote_props}")

            obstacle_properties = _majority_props(prop_responses, obstacle_keys)

        logger.info(f"[VLMGrounder-v6] P3 result: {obstacle_properties}")

        # --- Phase 4: Rule composition (N=1) ---
        composed_rules: List[Dict] = []
        composed_predicates: List[Dict] = []
        if obstacle_keys:
            # Build cleaned display names for rule composition
            disp_obs = []
            disp_props: Dict[str, List[str]] = {}
            for o in obstacle_keys:
                for cleaned, key in cleaned_map.items():
                    if key == o:
                        disp_obs.append(cleaned)
                        disp_props[cleaned] = obstacle_properties.get(o, [])
                        break

            disp_target = None
            if task_target_key:
                for cleaned, key in cleaned_map.items():
                    if key == task_target_key:
                        disp_target = cleaned
                        break
            disp_goal = None
            if goal_location_key:
                for cleaned, key in cleaned_map.items():
                    if key == goal_location_key:
                        disp_goal = cleaned
                        break

            p4_prompt = prompt_rule_composition(
                task_description, disp_target, disp_goal, disp_obs, disp_props, scene_desc
            )
            raw = client.infer(p4_prompt, images=[], max_tokens=512)
            parsed = _extract_json(raw)
            if parsed:
                raw_rules = parsed.get("rules", []) or []
                raw_composed = parsed.get("composed_predicates", []) or []
                # Translate display names in formulas back to obs keys for runtime
                composed_rules = _translate_rule_names(raw_rules, cleaned_map)
                composed_predicates = _translate_rule_names(raw_composed, cleaned_map, is_composed=True)
            logger.info(
                f"[VLMGrounder-v6] P4 result: {len(composed_rules)} rules, "
                f"{len(composed_predicates)} composed"
            )

        # Build obstacle radii from properties + composed rules
        obstacle_radii = _infer_radii(obstacle_keys, obstacle_properties, composed_rules)

        return {
            "obstacle_obs_keys": obstacle_keys,
            "task_target_obs_key": task_target_key,
            "goal_location_obs_key": goal_location_key,
            "obstacle_properties": obstacle_properties,
            "all_properties": obstacle_properties,
            "obstacle_radii": obstacle_radii,
            "confidence": 0.8 if obstacle_keys else 0.0,
            "method": "vlm-cot",
            "obstacle_obs_key": obstacle_keys[0] if obstacle_keys else None,
            "physical_properties": obstacle_properties.get(obstacle_keys[0], []) if obstacle_keys else [],
            "composed_rules": composed_rules,
            "composed_predicates": composed_predicates,
        }


# ---------------------------------------------------------------------------
# Rule name translation (display names → obs keys in formula strings)
# ---------------------------------------------------------------------------

def _translate_rule_names(
    rules: List[Dict],
    cleaned_map: Dict[str, str],
    is_composed: bool = False,
) -> List[Dict]:
    translated = []
    for rule in rules:
        r = dict(rule)
        key = "definition" if is_composed else "when"
        formula = r.get(key, "")
        if formula:
            for cleaned, obs_key in cleaned_map.items():
                # Replace cleaned name in formula with obs_key
                formula = re.sub(
                    re.escape(cleaned),
                    obs_key,
                    formula,
                    flags=re.IGNORECASE,
                )
            r[key] = formula
        translated.append(r)
    return translated


def _infer_radii(
    obstacles: List[str],
    properties: Dict[str, List[str]],
    rules: List[Dict],
) -> Dict[str, float]:
    radii: Dict[str, float] = {}
    for obs in obstacles:
        props = properties.get(obs, [])
        r = 0.18  # default
        if "IS_HOT" in props:
            r = 0.25
        elif "IS_FRAGILE" in props:
            r = 0.15
        elif "IS_SHARP" in props:
            r = 0.12
        # Try to read from composed rules
        for rule in rules:
            when = rule.get("when", "")
            then = rule.get("then", {})
            if obs in when and then.get("action") == "avoid":
                r = max(r, float(then.get("radius", r)))
        radii[obs] = r
    return radii


# ---------------------------------------------------------------------------
# Heuristic fallback (v3 behavior)
# ---------------------------------------------------------------------------

def _parse_target_heuristic(description: str, candidates: List[str]) -> Optional[str]:
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
    import re as _re
    for pattern in pickup_patterns:
        m = _re.search(pattern, desc)
        if m:
            phrase = m.group(1).strip().rstrip(".,;")
            best_match, best_score = None, 0
            for cand in candidates:
                cleaned = cand.replace("_", " ").rstrip(" 0123456789").replace(" obstacle", "").strip()
                words = cleaned.split()
                for start in range(len(words)):
                    suffix = " ".join(words[start:])
                    if suffix and suffix in phrase and len(suffix) > best_score:
                        best_match, best_score = cand, len(suffix)
            if best_match:
                return best_match
    return None


def _heuristic_result(candidates: List[str], parsed_target: Optional[str]) -> Dict:
    obs_key = next((n for n in candidates if "obstacle" in n.lower()), None)
    props: List[str] = []
    radius = 0.18
    if obs_key:
        n = obs_key.lower().replace("_", " ")
        if any(k in n for k in ["bowl", "cup", "glass", "mug", "pot"]):
            props.append("IS_SPILLABLE")
            radius = 0.20
        if any(k in n for k in ["egg", "vase", "crystal"]):
            props.append("IS_FRAGILE")
            radius = 0.15
        if any(k in n for k in ["candle", "torch", "flame", "heater", "stove"]):
            props.append("IS_HOT")
            radius = 0.25
    obs_list = [obs_key] if obs_key else []
    return {
        "obstacle_obs_keys": obs_list,
        "obstacle_obs_key": obs_key,
        "task_target_obs_key": parsed_target,
        "goal_location_obs_key": None,
        "obstacle_properties": {obs_key: props} if obs_key else {},
        "all_properties": {obs_key: props} if obs_key else {},
        "obstacle_radii": {obs_key: radius} if obs_key else {},
        "physical_properties": props,
        "confidence": 1.0 if obs_key else 0.0,
        "method": "heuristic",
        "composed_rules": [],
        "composed_predicates": [],
    }


def _empty_result() -> Dict:
    return {
        "obstacle_obs_keys": [],
        "obstacle_obs_key": None,
        "task_target_obs_key": None,
        "goal_location_obs_key": None,
        "obstacle_properties": {},
        "all_properties": {},
        "obstacle_radii": {},
        "physical_properties": [],
        "confidence": 0.0,
        "method": "heuristic",
        "composed_rules": [],
        "composed_predicates": [],
    }
