"""SafeLIBERO-specific utility functions for evaluation.

These functions override openvla-oft/libero_utils.py to properly handle
SafeLIBERO's camera configuration and avoid wrist camera contamination.
"""

import os
import math
import imageio
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from experiments.robot.robot_utils import DATE, DATE_TIME


def get_safelibero_env(task, model_family, resolution=256, include_wrist_camera=False,
                        camera_depths=False, camera_segmentations=None):
    """Initialize SafeLIBERO environment with explicit camera configuration.

    Args:
        task: LIBERO task object
        model_family: Model family name (for compatibility)
        resolution: Camera resolution (height and width)
        include_wrist_camera: If True, also enables robot0_eye_in_hand camera.
            Requires callers to copy observations immediately (np.ascontiguousarray)
            because both cameras share a single MuJoCo offscreen FBO.
            Safe at resolutions up to 1024 with hard_reset=False.
        camera_depths: If True, enables depth rendering; adds ``agentview_depth``
            to every observation dict. When False (default), ``agentview_depth``
            is absent and reading it raises KeyError. Set True together with
            ``camera_segmentations='instance'`` when use_safety_filter=True.
        camera_segmentations: Segmentation mode string (e.g. ``"instance"``); adds
            ``agentview_segmentation_instance`` to observations. Falsy values
            (None, "") are ignored and segmentation is disabled.

    Returns:
        env: Configured SafeLIBERO environment
        task_description: Natural language task description
    """
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    # Backview is excluded: OpenVLA was not trained on it (secondary=None in OXE config).
    # hard_reset=False: prevents EGL context recreation on each reset, which caused
    # grey-noise frame corruption when multiple cameras were active at high resolutions.
    camera_names = ["agentview", "robot0_eye_in_hand"] if include_wrist_camera else ["agentview"]
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_names": camera_names,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "hard_reset": False,
        "camera_depths": camera_depths,
    }
    if camera_segmentations:
        env_args["camera_segmentations"] = camera_segmentations

    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed affects object positions even with fixed initial state

    return env, task_description


def get_safelibero_image(obs, validate=True):
    """Extract and preprocess third-person camera image from SafeLIBERO obs.

    Args:
        obs: Observation dictionary from SafeLIBERO environment
        validate: If True, validates image for corruption (default: True)

    Returns:
        Preprocessed image (180° rotated, C-contiguous)

    Raises:
        ValueError: If image is corrupted or invalid
    """
    img = obs["agentview_image"]

    if validate:
        # Validate image to detect rendering failures (grey noise frames)
        if img is None or img.size == 0:
            raise ValueError("Invalid image: empty or None")

        # Check for corrupted frames using multiple heuristics:
        # 1. Extremely low variance (solid color) - rendering returned zeros/constant
        # 2. Uniform noise pattern - flat histogram indicates uninitialized memory/grey noise
        img_std = img.std()

        if img_std < 5.0:
            # Solid color or near-solid - definitely corrupted
            raise ValueError(f"Invalid image: solid color detected (std={img_std:.2f})")

        # Check for uniform noise (grey noise): histogram should not be too flat
        # Real scenes have structure - dark shadows, bright lights, varied colors create histogram peaks
        # Grey noise has uniform distribution - all histogram bins have similar counts (flat histogram)
        hist, _ = np.histogram(img.flatten(), bins=32, range=(0, 256))
        hist_std = hist.std()

        # Threshold: real scenes have hist_std > 100k typically, noise has hist_std < 10k
        # Adjusted based on image size (1024x1024x3 = ~3M pixels)
        hist_threshold = img.size / 300  # ~10k for 1024x1024x3 images

        if hist_std < hist_threshold:
            # Flat histogram = uniform noise = corrupted frame
            raise ValueError(f"Invalid image: uniform noise detected (hist_std={hist_std:.0f} < {hist_threshold:.0f})")

    # BUGFIX: Use ascontiguousarray to guarantee C-contiguous copy independent of simulator buffer
    # Rotation creates view with negative strides; ascontiguousarray ensures proper memory layout
    img = np.ascontiguousarray(img[::-1, ::-1])  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_safelibero_wrist_image(obs, validate=True):
    """Extract and preprocess the eye-in-hand camera image from SafeLIBERO obs.

    Only valid when the environment was created with include_wrist_camera=True.
    Applies the same 180° rotation as get_safelibero_image() to match training
    preprocessing (libero_utils.py get_libero_wrist_image does the same flip).

    Raises:
        ValueError: If image is corrupted or invalid (when validate=True)
    """
    img = obs["robot0_eye_in_hand_image"]

    if validate:
        if img is None or img.size == 0:
            raise ValueError("Invalid wrist image: empty or None")
        if img.std() < 5.0:
            raise ValueError(f"Invalid wrist image: solid color (std={img.std():.2f})")
        hist, _ = np.histogram(img.flatten(), bins=32, range=(0, 256))
        hist_threshold = img.size / 300
        if hist.std() < hist_threshold:
            raise ValueError(f"Invalid wrist image: uniform noise (hist_std={hist.std():.0f} < {hist_threshold:.0f})")

    return np.ascontiguousarray(img[::-1, ::-1])


def create_vlm_obs_env(task, resolution: int = 512):
    """Create secondary OffScreenRenderEnv for VLM observation capture.

    Enables camera_depths=True and camera_segmentations="instance" at
    `resolution` × `resolution`. Uses hard_reset=False to avoid EGL
    context recreation. Caller MUST call env.close() when done.

    Returns:
        env: Configured OffScreenRenderEnv
        task_description: str
    """
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    # backview omitted — unavailable in many task XMLs (see save_vlm_inputs.py:79-88)
    camera_names = ["agentview", "robot0_eye_in_hand"]
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_names": camera_names,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": True,
        "camera_segmentations": "instance",
        "hard_reset": False,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Save episode replay as MP4. Mirrors openvla-oft version but lives here for consistency."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = (
        task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    )
    mp4_path = (
        f"{rollout_dir}/{DATE_TIME}--openvla_oft--episode={idx}"
        f"--success={success}--task={processed_task_description}.mp4"
    )
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path
