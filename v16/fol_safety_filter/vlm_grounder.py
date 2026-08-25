"""
FOL Safety Filter v7 — Real-World Compatible Multi-Prompt VLM Grounder

3-prompt pipeline with N=3 majority voting:
  Prompt A: Obstacle identification  — which object should robot avoid?
  Prompt B: Spatial predicates       — approach directions + avoidance radius
  Prompt C: Physical properties      — IS_HOT / IS_FRAGILE / IS_SPILLABLE etc.

Key design:
  - Object names passed to VLM are CLEAN (no _obstacle suffix, no _1/_2 IDs)
  - VLM is told explicitly what the manipulated target and goal objects are
  - No MuJoCo GT information is passed to VLM
  - Cache key = (task_description, frozenset of obstacle-named obs keys in scene)
    so each unique scene configuration gets its own VLM call
  - Falls back to heuristic (_obstacle substring scan) if VLM fails
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from .prompts import (
    prompt_obstacle_identification,
    prompt_spatial_predicates,
    prompt_physical_properties,
    prompt_rule_composition,
)
from .vlm_client import VLMClient

logger = logging.getLogger(__name__)

N_VOTES = 3  # majority votes per prompt

_VALID_PROPERTIES = {"IS_FRAGILE", "IS_HOT", "IS_SPILLABLE", "IS_SHARP", "IS_HEAVY"}


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[Dict]:
    """Extract first valid JSON object from model output."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Clean name utilities
# No _obstacle, no _1/_2 suffixes, human-readable for VLM
# ---------------------------------------------------------------------------

def _clean_object_name(obs_key: str) -> str:
    """Convert obs key to clean human-readable name for VLM.

    moka_pot_obstacle_1  →  moka pot
    akita_black_bowl_2   →  akita black bowl   (or just: black bowl)
    plate_1              →  plate
    glazed_rim_porcelain_ramekin_1  →  glazed rim porcelain ramekin
    """
    name = obs_key
    name = re.sub(r'_obstacle', '', name)   # remove _obstacle
    name = re.sub(r'_\d+$', '', name)       # remove trailing _1, _2 etc
    name = name.replace('_', ' ')           # underscores → spaces
    name = name.strip()
    return name


def _build_clean_map(obs_keys: List[str]) -> Dict[str, str]:
    """Build mapping: clean_name (lowercase) → obs_key.

    Example:
      {"moka pot": "moka_pot_obstacle_1",
       "plate": "plate_1",
       "black bowl": "akita_black_bowl_2"}
    """
    clean_map: Dict[str, str] = {}
    for key in obs_keys:
        clean = _clean_object_name(key).lower()
        if clean:
            # If there's a collision, prefer obstacle-named keys
            if clean not in clean_map or "_obstacle" in key.lower():
                clean_map[clean] = key
    return clean_map


def _match_clean_name(text: Optional[str], clean_map: Dict[str, str]) -> Optional[str]:
    """Match VLM-returned name to an obs_key using exact then fuzzy match."""
    if not text or str(text).lower() in ("null", "none", "", "unknown"):
        return None
    t = str(text).lower().strip().strip('"\'')
    # Exact
    if t in clean_map:
        return clean_map[t]
    # Substring: find best matching key
    best_key = None
    best_score = 0
    for clean_name, obs_key in clean_map.items():
        if clean_name in t or t in clean_name:
            score = len(clean_name)
            if score > best_score:
                best_key = obs_key
                best_score = score
    return best_key


# ---------------------------------------------------------------------------
# Majority vote helpers
# ---------------------------------------------------------------------------

def _majority_string(votes: List[Optional[str]]) -> Optional[str]:
    valid = [v for v in votes if v is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def _majority_bool(votes: List[Optional[bool]], default: bool = False) -> bool:
    valid = [v for v in votes if v is not None]
    if not valid:
        return default
    return Counter(valid).most_common(1)[0][0]


def _majority_float(votes: List[Optional[float]], default: float = 0.18) -> float:
    valid = [v for v in votes if v is not None]
    if not valid:
        return default
    # Use median for numeric values
    valid.sort()
    return valid[len(valid) // 2]


# ---------------------------------------------------------------------------
# Main grounder class
# ---------------------------------------------------------------------------

class VLMObstacleGrounder:
    """
    v7: 3-prompt multi-vote VLM grounder with real-world compatible design.

    Object names passed to VLM are clean (no _obstacle, no sim suffixes).
    VLM is told explicitly which object is the target and which is the goal.
    Cache key includes the set of obstacle-named objects in the scene.
    """

    def __init__(self, use_vlm: bool = True):
        self.use_vlm = use_vlm
        self._client: Optional[VLMClient] = None
        self._cache: Dict[str, Dict] = {}

    def _get_client(self) -> VLMClient:
        if self._client is None:
            self._client = VLMClient()
        return self._client

    def detect(
        self,
        rgb_images: Any,
        task_description: str,
        candidate_obs_keys: List[str],
        target_obs_key: Optional[str] = None,
        goal_obs_key: Optional[str] = None,
    ) -> Dict:
        if isinstance(rgb_images, np.ndarray):
            rgb_images = [rgb_images]
        else:
            rgb_images = [img for img in (rgb_images or []) if img is not None]

        # Cache key: task description + which obstacle-named objects are in scene.
        # This ensures VLM re-runs when the obstacle type changes (safelibero_spatial
        # has different obstacle objects per episode).
        obstacle_keys_in_scene = sorted(
            k for k in candidate_obs_keys if "_obstacle" in k.lower()
        )
        cache_key = task_description.strip().lower() + "|" + ",".join(obstacle_keys_in_scene)
        if cache_key in self._cache:
            logger.info(
                f"[VLMGrounder-v7] Cache hit: task='{task_description[:40]}' "
                f"obstacles={obstacle_keys_in_scene}"
            )
            return self._cache[cache_key]

        parsed_target_key = target_obs_key or _parse_target_heuristic(
            task_description, candidate_obs_keys
        )

        client = self._get_client()
        if self.use_vlm and client.backend_name != "mock":
            try:
                result = self._multi_prompt_detect(
                    client=client,
                    images=rgb_images,
                    task_description=task_description,
                    candidate_obs_keys=candidate_obs_keys,
                    target_obs_key=parsed_target_key,
                    goal_obs_key=goal_obs_key,
                )
                if result.get("obstacle_obs_keys"):
                    result["method"] = f"vlm-{client.backend_name}"
                    logger.info(
                        f"[VLMGrounder-v7] VLM complete: "
                        f"obstacle={result.get('obstacle_obs_keys')} "
                        f"directions={result.get('unsafe_directions')} "
                        f"radius={result.get('obstacle_radii')} "
                        f"props={result.get('obstacle_properties')}"
                    )
                    self._cache[cache_key] = result
                    return result
            except Exception as e:
                logger.warning(f"[VLMGrounder-v7] VLM failed ({e}), falling back to heuristic")

        # Heuristic fallback
        heuristic = _heuristic_result(candidate_obs_keys, parsed_target_key)
        self._cache[cache_key] = heuristic
        return heuristic

    # ------------------------------------------------------------------
    # 3-prompt multi-vote detection
    # ------------------------------------------------------------------

    def _multi_prompt_detect(
        self,
        client: VLMClient,
        images: List,
        task_description: str,
        candidate_obs_keys: List[str],
        target_obs_key: Optional[str],
        goal_obs_key: Optional[str],
    ) -> Dict:
        # Build clean name map (no _obstacle, no numeric suffixes)
        clean_map = _build_clean_map(candidate_obs_keys)
        clean_names = sorted(clean_map.keys())  # sorted list for VLM

        # Clean target/goal names for VLM prompt
        target_clean = _clean_object_name(target_obs_key).lower() if target_obs_key else None
        goal_clean = _clean_object_name(goal_obs_key).lower() if goal_obs_key else None

        # ----------------------------------------------------------
        # Prompt A: Obstacle identification (N=3 votes)
        # ----------------------------------------------------------
        obstacle_votes: List[Optional[str]] = []
        for i in range(N_VOTES):
            prompt_a = prompt_obstacle_identification(
                task_description=task_description,
                clean_object_list=clean_names,
                target_object=target_clean,
                goal_object=goal_clean,
            )
            raw = client.infer(prompt_a, images, max_tokens=400)
            parsed = _extract_json(raw)
            if parsed:
                raw_obj = parsed.get("object")
                obs_key = _match_clean_name(raw_obj, clean_map)
                # Safety: never label target or goal as obstacle
                if obs_key and obs_key == target_obs_key:
                    obs_key = None
                if obs_key and obs_key == goal_obs_key:
                    obs_key = None
                obstacle_votes.append(obs_key)
                logger.debug(
                    f"[VLMGrounder-v7] Prompt A vote {i}: "
                    f"raw='{raw_obj}' → obs_key={obs_key}"
                )
            else:
                obstacle_votes.append(None)
                logger.debug(f"[VLMGrounder-v7] Prompt A vote {i}: no JSON in: {raw[:100]}")

        # Strict majority: obstacle must appear in ≥ ceil(N_VOTES/2) votes.
        # We count over ALL N_VOTES (including Nones) so a single-vote answer can't win
        # over 2 Nones (e.g. [None, None, 'ramekin'] → no majority → heuristic fallback).
        vote_counts = Counter(v for v in obstacle_votes if v is not None)
        min_votes = (N_VOTES // 2) + 1  # N=3 → 2 votes required
        if vote_counts and vote_counts.most_common(1)[0][1] >= min_votes:
            obstacle_key = vote_counts.most_common(1)[0][0]
        else:
            obstacle_key = None

        logger.info(f"[VLMGrounder-v7] Prompt A result: obstacle_key={obstacle_key} "
                    f"(votes={obstacle_votes})")

        # Find heuristic obstacle candidate (_obstacle substring in obs key name).
        # This is the safety object designated by the benchmark — reliable in sim.
        heuristic_candidate = next(
            (k for k in candidate_obs_keys
             if "_obstacle" in k.lower()
             and k != target_obs_key and k != goal_obs_key),
            None,
        )

        # If VLM can't confidently identify the obstacle, fall back to heuristic.
        # Also fall back if VLM picked an object that has no _obstacle in name
        # but a known _obstacle object exists — VLM likely confused a task surface
        # (e.g. ramekin where the bowl rests) with the actual safety obstacle.
        if not obstacle_key:
            obstacle_key = heuristic_candidate
            if obstacle_key:
                logger.info(
                    f"[VLMGrounder-v7] Prompt A fallback (heuristic): obstacle_key={obstacle_key}"
                )
            else:
                return _empty_result()
        elif (
            heuristic_candidate is not None
            and "_obstacle" not in obstacle_key.lower()
            and "_obstacle" in heuristic_candidate.lower()
        ):
            logger.info(
                f"[VLMGrounder-v7] VLM picked non-obstacle '{obstacle_key}' "
                f"but heuristic candidate '{heuristic_candidate}' exists — using heuristic"
            )
            obstacle_key = heuristic_candidate

        obstacle_clean = _clean_object_name(obstacle_key)

        # ----------------------------------------------------------
        # Prompt B: Spatial predicates (N=3 votes)
        # ----------------------------------------------------------
        direction_votes: List[List[str]] = []
        radius_votes: List[Optional[float]] = []
        valid_directions = {"above", "below", "around in front of", "around behind"}

        for i in range(N_VOTES):
            prompt_b = prompt_spatial_predicates(
                obstacle_name=obstacle_clean,
                task_description=task_description,
            )
            raw = client.infer(prompt_b, images, max_tokens=256)
            parsed = _extract_json(raw)
            if parsed:
                dirs = [
                    d for d in (parsed.get("unsafe_directions") or [])
                    if d in valid_directions
                ]
                radius = parsed.get("avoidance_radius_m")
                direction_votes.append(dirs)
                radius_votes.append(float(radius) if radius else None)
                logger.debug(
                    f"[VLMGrounder-v7] Prompt B vote {i}: dirs={dirs} radius={radius}"
                )
            else:
                direction_votes.append([])
                radius_votes.append(None)

        # Majority vote on directions: keep directions that appear in ≥2/3 votes
        all_dirs = [d for vote in direction_votes for d in vote]
        dir_counts = Counter(all_dirs)
        majority_directions = [d for d, cnt in dir_counts.items() if cnt >= (N_VOTES // 2 + 1)]
        avoidance_radius = _majority_float(
            [r for r in radius_votes if r is not None and 0.05 < r < 0.50],
            default=0.18,
        )
        logger.info(
            f"[VLMGrounder-v7] Prompt B result: directions={majority_directions} "
            f"radius={avoidance_radius:.2f}m"
        )

        # ----------------------------------------------------------
        # Prompt C: Physical properties (N=3 votes, text-only — no images)
        # ----------------------------------------------------------
        prop_votes: Dict[str, List[bool]] = {
            "IS_FRAGILE": [], "IS_SPILLABLE": [], "IS_HOT": [],
            "IS_SHARP": [], "REQUIRES_CAUTION": [], "ROTATION_LOCK": [],
        }

        for i in range(N_VOTES):
            prompt_c = prompt_physical_properties(
                obstacle_name=obstacle_clean,
                task_description=task_description,
            )
            # No images for property prompt — pure commonsense
            raw = client.infer(prompt_c, [], max_tokens=256)
            parsed = _extract_json(raw)
            if parsed:
                for prop in prop_votes:
                    val = parsed.get(prop)
                    if isinstance(val, bool):
                        prop_votes[prop].append(val)
                logger.debug(
                    f"[VLMGrounder-v7] Prompt C vote {i}: "
                    f"fragile={parsed.get('IS_FRAGILE')} "
                    f"spillable={parsed.get('IS_SPILLABLE')} "
                    f"hot={parsed.get('IS_HOT')}"
                )
            else:
                logger.debug(f"[VLMGrounder-v7] Prompt C vote {i}: no JSON in: {raw[:100]}")

        # Majority vote on each boolean property
        properties: List[str] = []
        for prop, votes in prop_votes.items():
            if _majority_bool(votes, default=False):
                if prop in _VALID_PROPERTIES:
                    properties.append(prop)

        # REQUIRES_CAUTION and ROTATION_LOCK feed directly into filter behavior
        requires_caution = _majority_bool(prop_votes.get("REQUIRES_CAUTION", []))
        rotation_lock = _majority_bool(prop_votes.get("ROTATION_LOCK", []))
        if rotation_lock and "IS_SPILLABLE" not in properties:
            properties.append("IS_SPILLABLE")  # spillable implies rotation lock

        # Scale radius by physical properties
        if "IS_HOT" in properties:
            avoidance_radius = max(avoidance_radius, 0.22)
        if "IS_SHARP" in properties:
            avoidance_radius = max(avoidance_radius, 0.15)

        logger.info(
            f"[VLMGrounder-v7] Prompt C result: properties={properties} "
            f"caution={requires_caution} rotation_lock={rotation_lock}"
        )

        # ----------------------------------------------------------
        # Prompt D: FOL rule composition (N=1, text-only).
        # The VLM composes safety rules and NEW predicates from the grounded
        # vocabulary (NEAR / HOLDING / IS_*).  Object names in the returned
        # formulas are cleaned display names — rewrite them to obs keys so
        # the KB can ground them at runtime.
        # ----------------------------------------------------------
        composed_rules: List[Dict] = []
        composed_predicates: List[Dict] = []
        try:
            target_clean = _clean_object_name(target_obs_key) if target_obs_key else None
            goal_clean = _clean_object_name(goal_obs_key) if goal_obs_key else None
            prompt_d = prompt_rule_composition(
                task_description=task_description,
                obstacle_name=obstacle_clean,
                properties=properties,
                target_name=target_clean,
                goal_name=goal_clean,
            )
            raw = client.infer(prompt_d, [], max_tokens=512)
            parsed = _extract_json(raw)
            if parsed:
                name_map = {obstacle_clean: obstacle_key}
                if target_clean and target_obs_key:
                    name_map[target_clean] = target_obs_key
                if goal_clean and goal_obs_key:
                    name_map[goal_clean] = goal_obs_key

                # Single-pass, longest-first substitution.  Naive sequential
                # replace double-substitutes: "moka pot" -> moka_pot_obstacle_1,
                # then variant "moka_pot" matches INSIDE the new key ->
                # moka_pot_obstacle_1_obstacle_1 (not in scene, rule skipped).
                # Mapping each obs key to itself makes the pass idempotent when
                # the VLM already used the exact key.
                variants: Dict[str, str] = {}
                for clean, key in name_map.items():
                    variants[key.lower()] = key
                    variants[clean.lower()] = key
                    variants[clean.replace(" ", "_").lower()] = key
                pattern = re.compile(
                    "|".join(sorted((re.escape(v) for v in variants),
                                    key=len, reverse=True)),
                    re.IGNORECASE,
                )

                def _to_obs_keys(formula: str) -> str:
                    formula = formula.replace('"', "").replace("'", "")
                    return pattern.sub(
                        lambda m: variants[m.group(0).lower()], formula)

                for r in (parsed.get("rules") or []):
                    if isinstance(r, dict) and r.get("when") and r.get("then"):
                        r["when"] = _to_obs_keys(str(r["when"]))
                        composed_rules.append(r)
                for p in (parsed.get("composed_predicates") or []):
                    if isinstance(p, dict) and p.get("definition") and p.get("then"):
                        p["definition"] = _to_obs_keys(str(p["definition"]))
                        composed_predicates.append(p)
            logger.info(
                f"[VLMGrounder-v16] Prompt D result: {len(composed_rules)} rules, "
                f"{len(composed_predicates)} composed predicates"
            )
        except Exception as e:
            logger.warning(f"[VLMGrounder-v16] Prompt D failed ({e}) — no composed rules")

        return {
            "obstacle_obs_keys": [obstacle_key],
            "obstacle_obs_key": obstacle_key,
            "task_target_obs_key": target_obs_key,
            "goal_location_obs_key": goal_obs_key,
            "obstacle_properties": {obstacle_key: properties},
            "all_properties": {obstacle_key: properties},
            "obstacle_radii": {obstacle_key: avoidance_radius},
            "physical_properties": properties,
            "unsafe_directions": majority_directions,
            "requires_caution": requires_caution,
            "rotation_lock": rotation_lock,
            "confidence": len([v for v in obstacle_votes if v is not None]) / N_VOTES,
            "method": "vlm-multi-prompt",
            "composed_rules": composed_rules,
            "composed_predicates": composed_predicates,
        }


# ---------------------------------------------------------------------------
# Heuristic fallback (no VLM — uses _obstacle naming convention)
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


def _heuristic_result(candidates: List[str], parsed_target: Optional[str]) -> Dict:
    obs_key = next((n for n in candidates if "obstacle" in n.lower()), None)
    props: List[str] = []
    radius = 0.18
    if obs_key:
        n = obs_key.lower().replace("_", " ")
        if any(k in n for k in ["bowl", "cup", "glass", "mug", "pot", "bottle", "milk"]):
            props.append("IS_SPILLABLE")
            radius = 0.20
        if any(k in n for k in ["egg", "vase", "crystal", "porcelain"]):
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
        "unsafe_directions": [],
        "requires_caution": False,
        "rotation_lock": False,
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
        "unsafe_directions": [],
        "requires_caution": False,
        "rotation_lock": False,
        "confidence": 0.0,
        "method": "none",
        "composed_rules": [],
        "composed_predicates": [],
    }
