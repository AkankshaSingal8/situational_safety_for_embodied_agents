"""Thin adapter around OopsieVerse's BEHAVIOR-1K (OmniGibson) task registry.

Wraps a single OopsieVerse B1K `OGDamageableEnvironment` for policy
evaluation, mirroring `robocasa_adapter.py`'s house style (same normalized
obs-dict contract idea, `get_health_summary()`, `close()`) but built on top
of `scripts/teleop_b1k.py`'s exact task-config -> env-config construction
path (imported, not reimplemented) -- see that module and
`b1k_headless_spike.py` for the confirmed-working construction pattern.

MUST run inside the `oopsieverse_b1k` conda env, INSIDE the
`b1k_container.sif` Apptainer container (see
`oopsieverse_eval/slurm/spike_b1k_headless.slurm` for the exact
`apptainer exec --nv` invocation this requires -- LD_LIBRARY_PATH
additions + Vulkan ICD binds are mandatory, not optional; bare metal on
this cluster cannot run Isaac Sim at all, see
`.superpowers/sdd/enchanted-finding-piglet/task-1-report.md`).

IMPORTANT: `OMNIGIBSON_HEADLESS=1` must be set in the environment (by the
launching shell / SLURM script) *before* this module (or anything that
transitively imports `omnigibson`) is imported -- `omnigibson.macros.gm`
locks `HEADLESS` on first read, which happens at `import omnigibson` time.
This module does not set the env var itself (unlike `b1k_headless_spike.py`,
which is a standalone script) because `run_b1k_pi05_eval.py` needs to set
it before importing anything, including this module.

One B1KAdapter instance is intended to be reused across many trials of the
SAME task within one process (construct once per SLURM array task, one
array element per B1K task name) -- NOT reconstructed per trial the way
`RoboCasaAdapter` is in `run_robocasa_pi05_eval.py`. Isaac Sim/Kit cold
boot took ~25 minutes in the confirmed-working headless spike (dominated
by first-run shader/extension cache population); reconstructing the env
per trial would make even a 3-trial smoke test impractically slow, let
alone a 30-trial full run.

Deviation from `RoboCasaAdapter`: reset() calls into
`teleop_b1k.reset_env(env, task_cfg, task_mod)` rather than a bare
`env.reset()`, because several task modules do essential runtime scene
setup in their own `task_mod.reset(env)` hook (e.g. `food_in_microwave`
seats the microwave onto the counter and opens its door there;
`wipe_counter` does runtime-only dirt/sponge setup) -- a bare `env.reset()`
would leave those tasks with a structurally broken/incomplete scene and
manufacture a false "0% success" independent of any policy's actual
behavior. `reset_env` also encodes a known GPU-dynamics-fluids gotcha
(re-running `env.reset()` on `pour_water`/`fill_bowl`/`turn_on_faucet`/
`nav_to_table`'s water-particle systems segfaults the renderer -- see its
docstring) by skipping its own retry-on-damaged-spawn loop for those
tasks. Reusing it here means this adapter automatically gets that
protection for free, without duplicating the gotcha's logic.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_OOPSIEVERSE_ROOT = os.path.abspath(os.path.join(_HERE, "..", "external", "oopsieverse"))
_SCRIPTS_DIR = os.path.join(_OOPSIEVERSE_ROOT, "scripts")
for _p in (_OOPSIEVERSE_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class B1KAdapter:
    """Thin wrapper around an OopsieVerse BEHAVIOR-1K `OGDamageableEnvironment`.

    Handles:
      - Looking up the task's `TaskConfig` via `teleop_b1k.load_task_config()`.
      - Constructing the underlying OmniGibson damageable env (external
        cameras always included -- needed for both the pi0.5 client's
        third-person view and video saving).
      - Normalizing `reset()`/`step()` observations into a stable dict
        contract (see `_build_obs()`).
      - Exposing per-(object, link) health values via `get_health_summary()`.

    Does NOT do any policy/VLA integration or video saving -- that's the
    caller's job (see `run_b1k_pi05_eval.py`), matching `RoboCasaAdapter`'s
    division of responsibility.
    """

    def __init__(
        self,
        task_name: str,
        external_image_size: int = 512,
        max_episode_steps: int = 700,
        seed: Optional[int] = None,
    ):
        # Imported here (not at module scope) so importing this module
        # doesn't require omnigibson to already be importable -- only
        # constructing an adapter instance does. `omnigibson` must already
        # be safely importable by the time this runs (OMNIGIBSON_HEADLESS
        # set by the caller before its own first `import omnigibson`).
        import omnigibson as og
        from omnigibson.macros import gm
        import teleop_b1k as tb1k

        self._og = og
        self._tb1k = tb1k

        self.task_name = task_name
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self._step_count = 0
        self._last_health: Optional[np.ndarray] = None
        self._env = None
        self._env_constructed = False

        self.task_cfg, self.task_mod = tb1k.load_task_config(task_name)

        # gm.USE_GPU_DYNAMICS / ENABLE_TRANSITION_RULES must be set before
        # the env is constructed (mirrors b1k_headless_spike.py /
        # teleop_b1k.main()'s exact sequencing).
        gm.USE_GPU_DYNAMICS = self.task_cfg.use_gpu_dynamics
        gm.ENABLE_TRANSITION_RULES = self.task_cfg.enable_transition_rules

        env_config = tb1k.build_env_config(self.task_cfg)
        if getattr(self.task_cfg, "external_camera_configs", None):
            env_config["env"]["external_sensors"] = tb1k.build_external_sensors_config(
                self.task_cfg, self.task_cfg.robot_name, self.task_cfg.robot_type,
                image_height=external_image_size, image_width=external_image_size,
            )

        from damagesim.omnigibson.damageable_env import OGDamageableEnvironment

        self._env = OGDamageableEnvironment(configs=env_config)
        self._env_constructed = True

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _nested_get(self, raw_obs: Dict[str, Any], *path: str) -> Optional[Any]:
        node = raw_obs
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node

    def _to_rgb_uint8(self, frame: Any) -> Optional[np.ndarray]:
        """RGBA (H,W,4) torch/np -> RGB (H,W,3) uint8 numpy, or None."""
        if frame is None:
            return None
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        frame = np.asarray(frame)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        return frame

    def _build_obs(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        robot_name = self.task_cfg.robot_name
        wrist_key_prefix = f"{robot_name}:eef_link:Camera:0"
        wrist_rgb = self._nested_get(raw_obs, robot_name, wrist_key_prefix, "rgb")

        ext0_rgb = self._nested_get(raw_obs, "external", "external_sensor0", "rgb")
        ext1_rgb = self._nested_get(raw_obs, "external", "external_sensor1", "rgb")

        proprio = self._nested_get(raw_obs, robot_name, "proprio")
        health = raw_obs.get("health") if isinstance(raw_obs, dict) else None
        eef_pos = raw_obs.get("eef_pos") if isinstance(raw_obs, dict) else None
        eef_ori = raw_obs.get("eef_ori") if isinstance(raw_obs, dict) else None

        def _to_np(x):
            if x is None:
                return None
            if hasattr(x, "cpu"):
                x = x.cpu().numpy()
            return np.asarray(x)

        obs: Dict[str, Any] = {
            "agentview_image": self._to_rgb_uint8(ext0_rgb),
            "secondary_image": self._to_rgb_uint8(ext1_rgb),
            "wrist_image": self._to_rgb_uint8(wrist_rgb),
            "proprio": _to_np(proprio),
            "eef_pos": _to_np(eef_pos),
            "eef_ori": _to_np(eef_ori),
            "gripper_qpos": self.get_gripper_qpos(),
            "health": _to_np(health),
            "raw": raw_obs,
        }
        self._last_health = obs["health"]
        return obs

    def get_gripper_qpos(self) -> Optional[np.ndarray]:
        """Return this robot's finger joint positions (per-DOF, typically
        2-dim for a Franka parallel-jaw gripper), read directly off
        `robot.get_joint_positions()` via the gripper controller's own
        `dof_idx` -- NOT a guessed slice offset into the flat `proprio`
        vector (per advisor guidance: avoid guessing indices into an
        opaque 24-dim proprio array when the robot object can report its
        own gripper DOF indices directly).
        """
        from b1k_action_utils import _find_arm_and_gripper_controller_names

        robot = self.robot
        try:
            _arm_name, gripper_name = _find_arm_and_gripper_controller_names(robot)
            dof_idx = robot.controller_joint_idx[gripper_name]
            if hasattr(dof_idx, "cpu"):
                dof_idx = dof_idx.cpu().numpy()
            qpos = robot.get_joint_positions()
            if hasattr(qpos, "cpu"):
                qpos = qpos.cpu().numpy()
            return np.asarray(qpos)[np.asarray(dof_idx)]
        except Exception:
            import traceback
            print("[B1KAdapter] WARNING: get_gripper_qpos() failed -- callers will silently see "
                  "np.zeros(2) as gripper state for EVERY step of EVERY episode. This is exactly "
                  "the silent-drop failure mode that produced a false 0% TSR on the RoboCasa side "
                  "before build_env_action's key bug was found -- do not trust results until fixed:")
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def env(self):
        return self._env

    @property
    def robot(self):
        return self._env.robots[0]

    def reset(self) -> Dict[str, Any]:
        """Reset the underlying env (via `teleop_b1k.reset_env`) and return
        the normalized obs dict.

        Uses `reset_env`, not a bare `env.reset()` -- see module docstring.

        `reset_env`'s FIRST action is `og.sim.viewer_camera.set_position_
        orientation(...)` -- a call `b1k_headless_spike.py` (which calls
        bare `env.reset()`) never exercised under `gm.HEADLESS=True`. If
        the viewer camera is absent/unusable headless, this would kill
        every single trial at reset. Falls back to the bare
        `env.reset()` + `task_mod.reset(env)` sequence (still gets the
        task-specific runtime scene setup, just skips the viewer-camera
        pose set) if `reset_env` raises, rather than let one headless
        quirk abort the whole 30-trial run.
        """
        self._step_count = 0
        try:
            self._tb1k.reset_env(self._env, self.task_cfg, self.task_mod)
        except Exception:
            import traceback
            print("[B1KAdapter] WARNING: teleop_b1k.reset_env() raised; falling back to bare "
                  "env.reset() + task_mod.reset(env) (skips viewer-camera pose set only):")
            traceback.print_exc()
            self._env.reset()
            if hasattr(self.task_mod, "reset") and callable(self.task_mod.reset):
                self.task_mod.reset(self._env)

        # reset_env doesn't return obs; grab the current obs by stepping
        # zero actions is wasteful/unsafe (would move the robot before we
        # even start) -- instead read the last obs off the env directly.
        #
        # IMPORTANT: must call `get_observation()` (OGDamageableEnvironment's
        # OWN override), NOT the base `Environment.get_obs()` inherited
        # unchanged -- confirmed by reading damagesim/omnigibson/
        # damageable_env.py directly: `eef_pos`/`eef_ori`/`health` are only
        # added by `_process_obs()`, which is called from the overridden
        # `reset()`/`step()` AND from `get_observation()` -- but NOT from
        # plain `get_obs()`. Calling `get_obs()` here (an earlier bug)
        # silently returned `eef_pos=None`/`eef_ori=None` for every
        # post-reset observation, corrupting the very first pi0.5 call of
        # every episode (and the diagnostic's own forced-motion test).
        raw_obs, _info = self._env.get_observation()
        obs = self._build_obs(raw_obs)

        # Blank-frame guard, matching b1k_headless_spike.py's own check
        # (frame.std() < 1.0 -> suspicious/likely-unrendered).
        img = obs.get("agentview_image")
        if img is not None:
            std = float(np.std(img))
            print(f"[B1KAdapter] post-reset agentview_image std={std:.3f} "
                  f"({'OK, looks rendered' if std >= 1.0 else 'SUSPICIOUS -- may be blank/unrendered'})")

        return obs

    def get_instruction(self) -> str:
        """Return a natural-language instruction for the task.

        B1K's `TaskConfig` (unlike RoboCasa's `get_ep_meta()`) has no
        language-instruction field at all, so this always falls back to a
        instruction derived from the task name -- this is the ONLY option
        here (not a fallback for an edge case, as in RoboCasaAdapter).
        """
        return self.task_name.replace("_", " ")

    def step(self, action) -> Tuple[Dict[str, Any], bool, bool, Dict[str, Any]]:
        """Step the underlying env with `action` (a torch tensor of shape
        (robot.action_dim,), per OmniGibson's `env.step()` contract --
        confirmed in `b1k_headless_spike.py`, which uses `th.zeros(...)`,
        NOT a numpy array like RoboCasa's robosuite `env.step()`)."""
        import torch as th

        if not isinstance(action, th.Tensor):
            action = th.as_tensor(np.asarray(action), dtype=th.float32)

        raw_obs, _reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1

        obs = self._build_obs(raw_obs)
        success = bool(
            hasattr(self.task_mod, "task_completion_check")
            and self.task_mod.task_completion_check(self._env)
        )
        episode_done = bool(terminated) or bool(truncated) or self._step_count >= self.max_episode_steps

        info = dict(info) if isinstance(info, dict) else {}
        info["health"] = obs["health"]

        return obs, success, episode_done, info

    def get_health_summary(self) -> Dict[str, float]:
        """Return {obj_link_name: health_value}, or {} if unavailable."""
        link_names = getattr(self._env, "health_list_link_names", None)
        if link_names is None or self._last_health is None:
            return {}
        return dict(zip(link_names, self._last_health.tolist()))

    def close(self) -> None:
        """Shut down the Kit SimulationApp.

        Unlike `RoboCasaAdapter.close()` (which is called once per trial,
        since a fresh robosuite env is constructed per trial), this should
        only be called ONCE, after ALL trials for this process's task are
        complete -- `og.shutdown()` tears down the whole Kit application,
        not just this env.
        """
        if self._og is not None:
            self._og.shutdown()
