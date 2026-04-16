"""SafeLIBERO-specific utility functions for evaluation.

These functions override openvla-oft/libero_utils.py to properly handle
SafeLIBERO's camera configuration and avoid wrist camera contamination.
"""

import os
import imageio
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def get_safelibero_env(task, model_family, resolution=256):
    """Initialize SafeLIBERO environment with explicit camera configuration.

    CRITICAL: Only enable agentview camera to prevent buffer conflicts
    with robot0_eye_in_hand that can cause frame contamination.

    Args:
        task: LIBERO task object
        model_family: Model family name (for compatibility)
        resolution: Camera resolution (height and width)

    Returns:
        env: Configured SafeLIBERO environment
        task_description: Natural language task description
    """
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    # BUGFIX: Explicitly set camera_names to ONLY agentview
    # Default SafeLIBERO includes ["agentview", "robot0_eye_in_hand", "backview"]
    # which can cause buffer conflicts at high resolutions
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_names": ["agentview"],  # ONLY third-person camera
        "camera_heights": resolution,
        "camera_widths": resolution,
    }

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
