"""Thin adapter around OopsieVerse's RoboCasa environment registry.

Wraps a single OopsieVerse RoboCasa `Damageable*` environment for policy
evaluation. Intended to run inside the `oopsieverse_robocasa` conda env
(Python 3.10, robosuite 1.5.2).

No policy/VLA logic here — this module only constructs the environment,
steps it with whatever action a caller provides, and normalizes the
observation dict into a stable contract. See
`oopsieverse_eval/.superpowers/sdd/task-3-brief.md` for the full spec.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Make the OopsieVerse submodule importable (oopsiebench.envs.registry, etc.)
# without modifying anything inside it.
_OOPSIEVERSE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "external", "oopsieverse"
)
_OOPSIEVERSE_ROOT = os.path.abspath(_OOPSIEVERSE_ROOT)
if _OOPSIEVERSE_ROOT not in sys.path:
    sys.path.insert(0, _OOPSIEVERSE_ROOT)


class RoboCasaAdapter:
    """Thin wrapper around an OopsieVerse RoboCasa `Damageable*` env.

    Handles:
      - Looking up the task's `EnvConfig` from `EnvironmentRegistry`.
      - Constructing the underlying robosuite/RoboCasa damageable env.
      - Normalizing `reset()`/`step()` observations into a stable dict
        contract (see class docstring / README for the key list).
      - Exposing per-(object, link) health values via `get_health_summary()`.

    Does NOT do any policy/VLA integration, video saving, or modification
    of the underlying env's behavior.
    """

    def __init__(
        self,
        task_name: str,
        camera_names: Optional[List[str]] = None,
        camera_width: int = 256,
        camera_height: int = 256,
        max_episode_steps: int = 400,
        seed: Optional[int] = None,
    ):
        # Imported here (rather than at module scope) so that importing this
        # module doesn't require the real oopsieverse_robocasa conda env to
        # be importable — only constructing an adapter instance does.
        from oopsiebench.envs.registry import EnvironmentRegistry
        from robosuite.controllers import load_composite_controller_config

        self.task_name = task_name
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self._step_count = 0
        self._last_health: Optional[np.ndarray] = None

        self.env_config = EnvironmentRegistry.get(task_name)

        if camera_names is None:
            default_cameras = [self.env_config.camera_name, "robot0_eye_in_hand"]
            # Deduplicate while preserving order.
            seen = set()
            camera_names = []
            for cam in default_cameras:
                if cam not in seen:
                    seen.add(cam)
                    camera_names.append(cam)
        self.camera_names = camera_names

        controller_configs = load_composite_controller_config(robot=self.env_config.robot)

        self.env = self.env_config.damageable_class(
            robots=self.env_config.robot,
            controller_configs=controller_configs,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=self.camera_names,
            camera_widths=camera_width,
            camera_heights=camera_height,
            camera_depths=False,
            ignore_done=True,
            control_freq=self.env_config.control_freq,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _safe_get(self, raw_obs: Dict[str, Any], key: str) -> Optional[Any]:
        """Look up `key` in `raw_obs`, returning None instead of raising."""
        try:
            return raw_obs[key]
        except (KeyError, TypeError):
            return None

    def _build_obs(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        agentview_key = f"{self.env_config.camera_name}_image"
        obs: Dict[str, Any] = {
            "agentview_image": self._safe_get(raw_obs, agentview_key),
            "eef_pos": self._safe_get(raw_obs, "robot0_eef_pos"),
            "eef_quat": self._safe_get(raw_obs, "robot0_eef_quat"),
            "gripper_qpos": self._safe_get(raw_obs, "robot0_gripper_qpos"),
            "health": self._safe_get(raw_obs, "health"),
            "raw": raw_obs,
        }

        wrist_image = self._safe_get(raw_obs, "robot0_eye_in_hand_image")
        if wrist_image is not None:
            obs["wrist_image"] = wrist_image

        # Only populated if the caller explicitly requested
        # "robot0_agentview_left" in camera_names (e.g. Cosmos-Policy-RoboCasa,
        # which hard-requires two third-person cameras) -- absent (None) for
        # every other policy client, which only ever request the default
        # [agentview_right, eye_in_hand] pair.
        secondary_image = self._safe_get(raw_obs, "robot0_agentview_left_image")
        if secondary_image is not None:
            obs["secondary_image"] = secondary_image

        self._last_health = obs["health"]
        return obs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """Reset the underlying env and return the normalized obs dict."""
        self._step_count = 0
        raw_obs, _info = self.env.reset()
        return self._build_obs(raw_obs)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], bool, bool, Dict[str, Any]]:
        """Step the underlying env with `action`, passed through unchanged."""
        raw_obs, _reward, done, info = self.env.step(action)
        self._step_count += 1

        obs = self._build_obs(raw_obs)
        success = self.env._check_success()
        episode_done = bool(done) or self._step_count >= self.max_episode_steps

        info = dict(info)
        info["health"] = obs["health"]

        return obs, success, episode_done, info

    def get_health_summary(self) -> Dict[str, float]:
        """Return {obj_link_name: health_value}, or {} if unavailable."""
        link_names = getattr(self.env, "health_list_link_names", None)
        if link_names is None or self._last_health is None:
            return {}
        return dict(zip(link_names, self._last_health))

    def close(self) -> None:
        """Close the underlying env if it defines `close()`, else no-op."""
        close_fn = getattr(self.env, "close", None)
        if callable(close_fn):
            close_fn()
