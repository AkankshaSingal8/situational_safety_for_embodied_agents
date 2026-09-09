"""libsafety_env_utils.py

Shared LIBERO-Safety environment/image helpers for the world-model-baseline
eval drivers (run_libsafety_cosmos_policy_eval.py, run_libsafety_fastwam_eval.py).

Mirrors vlm_pipeline/safelibero_utils.py (main repo, SafeLIBERO) but adapted for
LIBERO-Safety's per-task level/level_id bddl-file layout, matching the pattern
already established in run_libsafety_eval_openvla.py's get_libsafety_env().

NOTE: ControlEnv (LIBERO-Safety/libero/libero/envs/env_wrapper.py) defaults
camera_names to ["agentview", "robot0_eye_in_hand"], so the wrist camera is
already rendered without needing to pass camera_names explicitly.
"""

import pathlib

import numpy as np
from libero.libero.envs import OffScreenRenderEnv


def get_libsafety_env(task_suite, task, resolution=256):
    """Build the LIBERO env for a LIBERO-Safety task.

    Same as run_libsafety_eval_openvla.py's get_libsafety_env(): LIBERO-Safety
    nests an additional per-difficulty-level directory
    (<bddl_files>/<problem_folder>/L{level}/<bddl_file>), addressable only via
    Benchmark.get_task_bddl_file_path_by_level_id(level, level_id).
    """
    task_description = task.language
    task_bddl_file = task_suite.get_task_bddl_file_path_by_level_id(task.level, task.level_id)
    if task_bddl_file is None or not pathlib.Path(task_bddl_file).exists():
        raise FileNotFoundError(f"LIBERO-Safety bddl file not found for task {task}: {task_bddl_file}")
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed affects object positions even with fixed initial state
    return env, task_description


def _validate_image(img: np.ndarray, name: str = "image") -> None:
    if img is None or img.size == 0:
        raise ValueError(f"Invalid {name}: empty or None")
    img_std = img.std()
    if img_std < 5.0:
        raise ValueError(f"Invalid {name}: solid color detected (std={img_std:.2f})")
    hist, _ = np.histogram(img.flatten(), bins=32, range=(0, 256))
    hist_std = hist.std()
    hist_threshold = img.size / 300
    if hist_std < hist_threshold:
        raise ValueError(f"Invalid {name}: uniform noise detected (hist_std={hist_std:.0f} < {hist_threshold:.0f})")


def get_libsafety_image(obs, validate=True):
    """Extract + 180°-rotate the third-person camera image (matches training preprocessing)."""
    img = obs["agentview_image"]
    if validate:
        _validate_image(img, "image")
    return np.ascontiguousarray(img[::-1, ::-1])


def get_libsafety_wrist_image(obs, validate=True):
    """Extract + 180°-rotate the eye-in-hand camera image."""
    img = obs["robot0_eye_in_hand_image"]
    if validate:
        _validate_image(img, "wrist image")
    return np.ascontiguousarray(img[::-1, ::-1])
