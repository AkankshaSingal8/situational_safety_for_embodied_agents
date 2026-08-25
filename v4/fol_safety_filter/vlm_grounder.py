"""
VLM Obstacle Grounder — v4

At episode start, identifies which object in the scene is the safety-relevant
obstacle. Two modes:

  heuristic (default): scan candidate names for "obstacle" substring — same
    as the eval script's joint-name scan, but operates on obs dict keys.

  vlm (QWEN_SERVER_URL set): POST to /infer_fol on the Qwen inference server.
    Falls back to heuristic if the server is unavailable or returns no match.

detect() now accepts a list of RGB images (agentview + eye_in_hand).
"""

from __future__ import annotations

import base64
import io
import logging
import os
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VLMObstacleGrounder:
    """
    Identifies the obstacle and physical properties from scene images.

    Returns a predicate_dict:
    {
      "obstacle_obs_key": str | None,   # matched key in obs dict (no _pos suffix)
      "physical_properties": List[str], # e.g. ["IS_SPILLABLE", "IS_FRAGILE"]
      "confidence": float,              # 0.0–1.0
      "method": str,                    # "vlm" | "heuristic"
    }
    """

    def __init__(self, use_vlm: bool = False, n_votes: int = 3):
        self.use_vlm = use_vlm
        self.n_votes = n_votes
        self._last_predicate_dict: Dict = {}

    def detect(
        self,
        rgb_images,  # np.ndarray OR List[np.ndarray]
        task_description: str,
        candidate_obs_keys: List[str],
    ) -> Dict:
        """
        Main entry point. Returns predicate_dict.

        rgb_images: single image array OR list of image arrays
            (agentview first, then eye_in_hand).
        candidate_obs_keys: list of object names from obs (no _pos suffix),
            e.g. ["moka_pot_obstacle_1", "plate_1", "akita_black_bowl_1"]
        """
        # Normalize to list
        if isinstance(rgb_images, np.ndarray):
            rgb_images = [rgb_images]
        else:
            rgb_images = [img for img in (rgb_images or []) if img is not None]

        result = {
            "obstacle_obs_key": None,
            "physical_properties": [],
            "confidence": 0.0,
            "method": "heuristic",
        }

        server_url = os.environ.get("QWEN_SERVER_URL", "").strip()
        should_use_vlm = self.use_vlm and bool(server_url) and bool(rgb_images)

        if should_use_vlm:
            try:
                vlm_result = self._vlm_detect(rgb_images, task_description, candidate_obs_keys, server_url)
                if vlm_result.get("obstacle_obs_key"):
                    result.update(vlm_result)
                    result["method"] = "vlm"
                    self._last_predicate_dict = result
                    return result
                else:
                    logger.info("[VLMGrounder] VLM returned no match, falling back to heuristic")
            except Exception as e:
                logger.warning(f"[VLMGrounder] VLM detection failed ({e}), falling back to heuristic")

        # Heuristic fallback
        key = self._heuristic_detect(candidate_obs_keys)
        result["obstacle_obs_key"] = key
        result["confidence"] = 1.0 if key else 0.0
        result["physical_properties"] = self._heuristic_properties(key) if key else []
        self._last_predicate_dict = result
        return result

    # ------------------------------------------------------------------
    # Heuristic detection
    # ------------------------------------------------------------------

    def _heuristic_detect(self, candidates: List[str]) -> Optional[str]:
        """Return first candidate whose name contains 'obstacle'."""
        for name in candidates:
            if "obstacle" in name.lower():
                return name
        return None

    def _heuristic_properties(self, obs_key: str) -> List[str]:
        """Infer physical properties from object name keywords."""
        props = []
        n = obs_key.lower().replace("_", " ")
        if any(k in n for k in ["bowl", "cup", "glass", "mug", "pot", "container"]):
            props.append("IS_SPILLABLE")
        if any(k in n for k in ["egg", "vase", "crystal"]):
            props.append("IS_FRAGILE")
        if any(k in n for k in ["candle", "torch", "flame", "heater", "stove"]):
            props.append("IS_HOT")
        return props

    # ------------------------------------------------------------------
    # VLM detection via HTTP /infer_fol endpoint
    # ------------------------------------------------------------------

    def _vlm_detect(
        self,
        rgb_images: List[np.ndarray],
        task_description: str,
        candidates: List[str],
        server_url: str,
    ) -> Dict:
        """
        POST to QWEN_SERVER_URL/infer_fol. Returns partial predicate_dict.
        Raises on HTTP error or timeout.
        """
        import requests
        from PIL import Image

        # Encode images as base64 PNG
        images_b64 = []
        for img in rgb_images:
            pil_img = Image.fromarray(img.astype(np.uint8))
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        # Build cleaned candidates: "moka_pot_obstacle_1" → "moka pot"
        cleaned_map: Dict[str, str] = {}  # cleaned_name → obs_key
        for name in candidates:
            cleaned = name.replace("_", " ").rstrip(" 0123456789").strip()
            # Remove trailing "obstacle" label so VLM doesn't cheat
            cleaned = cleaned.replace(" obstacle", "").strip()
            if cleaned:
                cleaned_map[cleaned] = name

        payload = {
            "images_b64": images_b64,
            "task_description": task_description,
            "candidates": list(cleaned_map.keys()),
            "num_votes": self.n_votes,
        }

        resp = requests.post(
            f"{server_url.rstrip('/')}/infer_fol",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        obstacle_cleaned = data.get("obstacle", "none") or "none"
        obstacle_obs_key = None

        if obstacle_cleaned.lower() != "none":
            # Exact match first
            obstacle_obs_key = cleaned_map.get(obstacle_cleaned)
            # Fuzzy match fallback
            if obstacle_obs_key is None:
                for cleaned, obs_key in cleaned_map.items():
                    if cleaned.lower() in obstacle_cleaned.lower() or obstacle_cleaned.lower() in cleaned.lower():
                        obstacle_obs_key = obs_key
                        break

        logger.info(
            f"[VLMGrounder] Server obstacle: '{obstacle_cleaned}' → matched obs_key: {obstacle_obs_key}"
            f" | props={data.get('physical_properties', [])}"
            f" | confidence={data.get('confidence', 0):.2f}"
        )

        return {
            "obstacle_obs_key": obstacle_obs_key,
            "physical_properties": data.get("physical_properties", []),
            "confidence": data.get("confidence", 0.0),
        }

    def _match_to_candidate(self, vlm_text: str, cleaned_map: Dict[str, str]) -> Optional[str]:
        """Longest-substring match between VLM output and cleaned candidate names."""
        best_key = None
        best_score = 0
        for cleaned, obs_key in cleaned_map.items():
            if cleaned.lower() in vlm_text and len(cleaned) > best_score:
                best_key = obs_key
                best_score = len(cleaned)
        return best_key


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _majority_text(responses: List[str]) -> str:
    """Return most common response string (case-insensitive majority vote)."""
    normalized = [r.lower().strip() for r in responses]
    if not normalized:
        return ""
    most_common, _ = Counter(normalized).most_common(1)[0]
    return most_common
