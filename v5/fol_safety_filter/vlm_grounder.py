"""
VLM Grounder v5 — Structured single-call obstacle + property detection.

Calls /infer_fol_v5 which returns task_target, goal_location, obstacles,
all_properties in one structured JSON response.
Falls back to heuristic (v3 behavior) when server unavailable.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _parse_target_from_description(description: str, candidates: List[str]) -> Optional[str]:
    """Parse the pickup target from a task description without GT info."""
    if not description:
        return None
    desc = description.lower()

    pickup_patterns = [
        r"pick up (?:the )?(.+?)(?:\s+(?:between|on|from|in|and|that|which)|$)",
        r"grab (?:the )?(.+?)(?:\s+(?:between|on|from|in|and|that|which)|$)",
        r"grasp (?:the )?(.+?)(?:\s+(?:between|on|from|in|and|that|which)|$)",
        r"get (?:the )?(.+?)(?:\s+(?:between|on|from|in|and|that|which)|$)",
        r"lift (?:the )?(.+?)(?:\s+(?:between|on|from|in|and|that|which)|$)",
        r"move (?:the )?(.+?) (?:to|onto|into)",
        r"put (?:the )?(.+?) (?:on|in|into|onto)",
        r"place (?:the )?(.+?) (?:on|in|into|onto)",
    ]

    for pattern in pickup_patterns:
        m = re.search(pattern, desc)
        if m:
            target_phrase = m.group(1).strip().rstrip(".,;")
            best_match = None
            best_score = 0
            for cand in candidates:
                cleaned = cand.replace("_", " ").rstrip(" 0123456789").strip()
                cleaned = cleaned.replace(" obstacle", "").strip()
                if not cleaned:
                    continue
                # Match if: cleaned in phrase, or any word-sequence of cleaned is in phrase
                # This handles "akita black bowl" matching phrase "black bowl between..."
                words = cleaned.split()
                # Try all suffixes of the cleaned name (longest first)
                matched = False
                for start in range(len(words)):
                    suffix = " ".join(words[start:])
                    if suffix and suffix in target_phrase and len(suffix) > best_score:
                        best_match = cand
                        best_score = len(suffix)
                        matched = True
                        break
            if best_match:
                return best_match
    return None


class VLMObstacleGrounder:
    """
    v5: Identifies obstacles, task target, goal location, and all properties
    from a SINGLE structured VLM call.

    Returns a predicate_dict:
    {
      "obstacle_obs_keys": List[str],
      "task_target_obs_key": str | None,
      "goal_location_obs_key": str | None,
      "obstacle_properties": {obs_key: List[str], ...},
      "all_properties": {obs_key: List[str], ...},
      "obstacle_radii": {obs_key: float, ...},
      "confidence": float,
      "method": "vlm" | "heuristic",
      # backward compat
      "obstacle_obs_key": str | None,
      "physical_properties": List[str],
    }
    """

    def __init__(self, use_vlm: bool = True, n_votes: int = 1):
        self.use_vlm = use_vlm
        self.n_votes = n_votes

    def detect(
        self,
        rgb_images,
        task_description: str,
        candidate_obs_keys: List[str],
    ) -> Dict:
        if isinstance(rgb_images, np.ndarray):
            rgb_images = [rgb_images]
        else:
            rgb_images = [img for img in (rgb_images or []) if img is not None]

        result = {
            "obstacle_obs_keys": [],
            "task_target_obs_key": None,
            "goal_location_obs_key": None,
            "obstacle_properties": {},
            "all_properties": {},
            "obstacle_radii": {},
            "confidence": 0.0,
            "method": "heuristic",
            "obstacle_obs_key": None,
            "physical_properties": [],
        }

        server_url = os.environ.get("QWEN_SERVER_URL", "").strip()
        should_use_vlm = self.use_vlm and bool(server_url) and bool(rgb_images)

        parsed_target_key = _parse_target_from_description(task_description, candidate_obs_keys)
        parsed_target_cleaned = None
        if parsed_target_key:
            parsed_target_cleaned = parsed_target_key.replace("_", " ").rstrip(" 0123456789").strip()
            parsed_target_cleaned = parsed_target_cleaned.replace(" obstacle", "").strip()
            logger.info(f"[VLMGrounder-v5] Parsed target from description: '{parsed_target_cleaned}' (key={parsed_target_key})")

        if should_use_vlm:
            try:
                vlm_result = self._vlm_detect(
                    rgb_images, task_description, candidate_obs_keys,
                    server_url, parsed_target_cleaned,
                )
                if vlm_result.get("obstacle_obs_keys") is not None:
                    result.update(vlm_result)
                    result["method"] = "vlm"
                    if vlm_result["obstacle_obs_keys"]:
                        first_obs = vlm_result["obstacle_obs_keys"][0]
                        result["obstacle_obs_key"] = first_obs
                        result["physical_properties"] = vlm_result["obstacle_properties"].get(first_obs, [])
                    logger.info(
                        f"[VLMGrounder-v5] VLM: target={vlm_result.get('task_target_obs_key')} "
                        f"goal={vlm_result.get('goal_location_obs_key')} "
                        f"obstacles={vlm_result.get('obstacle_obs_keys')} "
                        f"conf={vlm_result.get('confidence', 0):.2f}"
                    )
                    return result
                else:
                    logger.info("[VLMGrounder-v5] VLM returned no obstacles, falling back to heuristic")
            except Exception as e:
                logger.warning(f"[VLMGrounder-v5] VLM failed ({e}), falling back to heuristic")

        # Heuristic fallback
        heuristic_key = self._heuristic_detect(candidate_obs_keys)
        if heuristic_key:
            result["obstacle_obs_keys"] = [heuristic_key]
            result["obstacle_obs_key"] = heuristic_key
            props = self._heuristic_properties(heuristic_key)
            result["obstacle_properties"] = {heuristic_key: props}
            result["physical_properties"] = props
            result["obstacle_radii"] = {heuristic_key: self._default_radius(props)}
            result["confidence"] = 1.0
        if parsed_target_key:
            result["task_target_obs_key"] = parsed_target_key
        logger.info(
            f"[VLMGrounder-v5] Heuristic: obstacle={result['obstacle_obs_key']} "
            f"target={result['task_target_obs_key']}"
        )
        return result

    def _heuristic_detect(self, candidates: List[str]) -> Optional[str]:
        for name in candidates:
            if "obstacle" in name.lower():
                return name
        return None

    def _heuristic_properties(self, obs_key: str) -> List[str]:
        props = []
        n = obs_key.lower().replace("_", " ")
        if any(k in n for k in ["bowl", "cup", "glass", "mug", "pot", "container"]):
            props.append("IS_SPILLABLE")
        if any(k in n for k in ["egg", "vase", "crystal"]):
            props.append("IS_FRAGILE")
        if any(k in n for k in ["candle", "torch", "flame", "heater", "stove"]):
            props.append("IS_HOT")
        return props

    def _default_radius(self, props: List[str]) -> float:
        if "IS_HOT" in props:
            return 0.25
        if "IS_SHARP" in props:
            return 0.15
        if "IS_FRAGILE" in props:
            return 0.12
        if "IS_SPILLABLE" in props:
            return 0.18
        return 0.18

    def _vlm_detect(
        self,
        rgb_images: List[np.ndarray],
        task_description: str,
        candidates: List[str],
        server_url: str,
        parsed_target_cleaned: Optional[str],
    ) -> Dict:
        import requests
        from PIL import Image

        images_b64 = []
        for img in rgb_images:
            pil_img = Image.fromarray(img.astype(np.uint8))
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        cleaned_map: Dict[str, str] = {}
        for name in candidates:
            cleaned = name.replace("_", " ").rstrip(" 0123456789").strip()
            cleaned = cleaned.replace(" obstacle", "").strip()
            if cleaned:
                cleaned_map[cleaned] = name

        payload = {
            "images_b64": images_b64,
            "task_description": task_description,
            "candidates": list(cleaned_map.keys()),
            "parsed_target": parsed_target_cleaned,
        }

        resp = requests.post(
            f"{server_url.rstrip('/')}/infer_fol_v5",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()

        def match_key(name_str: Optional[str]) -> Optional[str]:
            if not name_str or str(name_str).lower() in ("null", "none", ""):
                return None
            if name_str in cleaned_map:
                return cleaned_map[name_str]
            for cleaned, obs_key in cleaned_map.items():
                if cleaned.lower() in name_str.lower() or name_str.lower() in cleaned.lower():
                    return obs_key
            return None

        task_target_key = match_key(data.get("task_target"))
        goal_location_key = match_key(data.get("goal_location"))

        obstacle_obs_keys = []
        obstacle_properties = {}
        obstacle_radii = {}
        for obs_entry in data.get("obstacles", []):
            name_str = obs_entry.get("name", "")
            obs_key = match_key(name_str)
            if not obs_key:
                continue
            if obs_key == task_target_key or obs_key == goal_location_key:
                logger.info(f"[VLMGrounder-v5] Skipping VLM obstacle '{obs_key}' — matches target or goal")
                continue
            obstacle_obs_keys.append(obs_key)
            props = obs_entry.get("properties", [])
            obstacle_properties[obs_key] = props
            radius = float(obs_entry.get("avoidance_radius", 0.18))
            if "IS_HOT" in props:
                radius = min(radius, 0.28)
            elif "IS_FRAGILE" in props:
                radius = min(radius, 0.15)
            elif "IS_SHARP" in props:
                radius = min(radius, 0.18)
            else:
                radius = min(radius, 0.22)
            obstacle_radii[obs_key] = max(0.10, radius)

        all_properties = {}
        for cleaned_name, props in data.get("all_properties", {}).items():
            obs_key = match_key(cleaned_name)
            if obs_key:
                all_properties[obs_key] = props

        return {
            "obstacle_obs_keys": obstacle_obs_keys,
            "task_target_obs_key": task_target_key,
            "goal_location_obs_key": goal_location_key,
            "obstacle_properties": obstacle_properties,
            "all_properties": all_properties,
            "obstacle_radii": obstacle_radii,
            "confidence": float(data.get("confidence", 0.7)),
        }
