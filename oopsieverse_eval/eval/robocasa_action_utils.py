"""Shared helpers for turning a 7-dim VLA action (delta EEF pose + gripper)
into a full OopsieVerse/RoboCasa composite-controller action vector.

Used by both the OpenVLA and pi0.5 RoboCasa eval clients so the two share
identical arm/gripper/base wiring and only differ in how they call their
respective policy server.
"""

from __future__ import annotations

import numpy as np


def json_default(obj):
    """`json.dump(..., default=json_default)` handler for numpy scalar types.

    `RoboCasaAdapter.get_health_summary()` returns numpy float32 values,
    which the stdlib `json` module doesn't know how to serialize.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def build_env_action(robot, arm_delta: np.ndarray, gripper_value: float) -> np.ndarray:
    """Build a full composite-controller action vector for `robot`.

    `arm_delta` is a 6-dim [dx, dy, dz, d_roll, d_pitch, d_yaw] OSC-delta
    command, clipped to [-1, 1] (the composite controller's expected range).
    `gripper_value` follows RoboCasa/robosuite convention: -1.0 = open,
    +1.0 = closed.

    For mobile robots (e.g. PandaOmron), the base is held still (`base_mode`
    set to "arm mode") -- OpenVLA/pi0.5 have no native base-control output,
    so navigation-dependent tasks are expected to be out of reach for an
    off-the-shelf single-arm-tuned policy; this is a known limitation, not a
    bug in this adapter.
    """
    arm = robot.arms[0]
    gripper_dof = robot.gripper[arm].dof

    action_dict = {
        f"{arm}_delta": np.clip(np.asarray(arm_delta, dtype=np.float64), -1.0, 1.0),
        f"{arm}_gripper": np.array([gripper_value] * gripper_dof),
    }
    if robot.is_mobile:
        action_dict["base"] = np.zeros(3)
        action_dict["base_mode"] = np.array([-1])

    return robot.create_action_vector(action_dict)


def openvla_gripper_to_robocasa(raw_gripper: float) -> float:
    """Convert OpenVLA's training-time gripper convention (0=close, 1=open)
    to RoboCasa/robosuite's (-1=open, +1=close).

    Mirrors the two-step normalize-then-invert conversion used by the
    existing openvla-oft SafeLIBERO pipeline in this repo (reimplemented
    here, not imported, to keep this build fully independent).
    """
    normalized = raw_gripper * 2.0 - 1.0  # [0,1] -> [-1,1], -1=close,+1=open
    binarized = 1.0 if normalized > 0 else -1.0  # binarize like normalize_gripper_action(binarize=True)
    return -binarized  # invert: robosuite convention is -1=open,+1=close
