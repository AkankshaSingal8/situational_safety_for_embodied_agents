"""Shared helpers for turning a 7-dim VLA action (delta EEF pose + gripper)
into a full OmniGibson/B1K composite-controller action tensor.

Mirrors `robocasa_action_utils.py`'s DESIGN PATTERN (a `build_env_action`
that assembles a per-controller action dict/vector from a robot-agnostic
arm-delta + gripper command), NOT its code -- B1K's controller stack is a
genuinely different convention from RoboCasa/robosuite's composite
controller, confirmed directly from OmniGibson source (NOT guessed):

  - Arm: `InverseKinematicsController`, mode "pose_delta_ori" (the class
    default -- no task config in `oopsiebench/envs/behavior1k/*.py`
    overrides `mode`), with `command_input_limits=None`. Per
    `ik_controller.py`'s own docstring: "If both [command_input_limits]
    and [command_output_limits] is not None, then all inputted command
    values will be scaled... If EITHER is None, no scaling will be used."
    With `command_input_limits=None`, the 6-dim command is used AS-IS,
    directly as (dx,dy,dz) in METERS and (dax,day,daz) axis-angle deltas
    in RADIANS, relative to the robot base frame -- there is no [-1,1]
    normalization step at all on the B1K side, unlike RoboCasa's OSC
    controller. pi0.5 (trained on LIBERO/robosuite data) emits actions in
    robosuite's normalized OSC-delta convention, so this module must apply
    an explicit physical scale factor before handing the command to B1K's
    IK controller -- passing pi0.5's raw [-1,1]-ish output straight through
    would command up to a full 1-meter position jump per control step.
    `ARM_POS_SCALE_M` / `ARM_ROT_SCALE_RAD` below encode LIBERO/robosuite's
    own standard per-step OSC_POSE delta convention (see docstring on those
    constants) as the scale factor -- this is what pi0.5's raw output
    numerically represents, since it was trained on exactly that data.
  - Gripper: `MultiFingerGripperController`, mode "smooth" (set explicitly
    in every B1K task config read), `command_input_limits=(0.0, 1.0)`.
    Command is a single scalar in [0, 1], symmetric across both fingers;
    NOT robosuite's [-1,1] convention. `smooth` mode's default output
    range is the mean of the gripper's own independent joint limits, so
    which end of [0, 1] means "open" vs "closed" depends on which
    direction increasing finger-joint position goes for this robot's
    gripper -- assumed here to be the standard Franka convention (0 =
    closed / fingers together, 1 = open / fingers apart), matching every
    B1K task's Franka-family robot (FrankaPanda or FrankaMounted). THIS
    ASSUMPTION MUST BE CONFIRMED empirically before trusting any grasp
    result -- see `b1k_introspect_and_smoke.py`'s forced-gripper-command
    diagnostic, run once as part of the first GPU job for this phase.

Robot-agnostic controller wiring: instead of hardcoding "arm_0"/"gripper_0"
(true for every single-arm Franka B1K task, but NOT for `nav_to_table`,
whose robot is a dual-arm mobile Tiago with `arm_left`/`gripper_left`/
`arm_right`/`gripper_right` controllers -- confirmed by reading its task
config), this module looks up the actual controller names from
`robot.controller_action_idx` (confirmed to exist on OmniGibson's
`ControllableObject` base class, `objects/controllable_object.py`) and
drives whichever arm/gripper pair is alphabetically first. Any other
controllers (a second arm, a mobile base, a trunk) are left at a zero
no-op command -- the exact same "hold everything else still" strategy
`robocasa_action_utils.py` already documents for RoboCasa's mobile
PandaOmron (OpenVLA/pi0.5 have no native base or second-arm output), so
`nav_to_table` results should be read with that limitation in mind.
"""

from __future__ import annotations

import numpy as np


def json_default(obj):
    """`json.dump(..., default=json_default)` handler for numpy scalar types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# LIBERO/robosuite's standard OSC_POSE controller default per-step output
# range (`output_max`/`output_min` in robosuite's default OSC_POSE
# controller config, the convention pi0.5/pi05_libero's raw [-1,1] action
# outputs are trained against): position delta up to 5cm/step, rotation
# delta up to ~0.5 rad/step. Used here to convert pi0.5's raw normalized
# output into the physical meters/radians B1K's `command_input_limits=None`
# IK controller expects verbatim.
ARM_POS_SCALE_M = 0.05
ARM_ROT_SCALE_RAD = 0.5


def _find_arm_and_gripper_controller_names(robot) -> tuple[str, str]:
    """Return (arm_controller_name, gripper_controller_name) for the first
    arm/gripper pair in `robot.controller_action_idx`, e.g. ("arm_0",
    "gripper_0") for every single-arm Franka B1K task, or ("arm_left",
    "gripper_left") for the dual-arm Tiago used by `nav_to_table` (picks
    the alphabetically-first arm; the other arm/base/trunk are held at a
    zero no-op command by the caller).
    """
    idx_map = robot.controller_action_idx
    arm_names = sorted(k for k in idx_map.keys() if k.startswith("arm"))
    if not arm_names:
        raise ValueError(f"No 'arm*' controller found in controller_action_idx: {list(idx_map.keys())}")
    arm_name = arm_names[0]
    gripper_name = arm_name.replace("arm", "gripper", 1)
    if gripper_name not in idx_map:
        raise ValueError(
            f"Expected matching gripper controller '{gripper_name}' for arm '{arm_name}', "
            f"got controllers: {list(idx_map.keys())}"
        )
    return arm_name, gripper_name


def build_b1k_env_action(
    robot,
    arm_delta: np.ndarray,
    gripper_value: float,
    arm_pos_scale: float = ARM_POS_SCALE_M,
    arm_rot_scale: float = ARM_ROT_SCALE_RAD,
):
    """Build a full OmniGibson action tensor (shape (robot.action_dim,)) for `robot`.

    `arm_delta` is a 6-dim [dx, dy, dz, d_roll, d_pitch, d_yaw] raw pi0.5
    output, in robosuite/LIBERO's normalized OSC-delta convention (roughly
    [-1, 1] per component). Rescaled here into meters/radians via
    `arm_pos_scale`/`arm_rot_scale` before being passed straight through to
    B1K's `command_input_limits=None` IK controller (see module docstring
    -- no further clipping/scaling happens on the B1K side).

    `gripper_value` follows B1K's own `smooth`-mode convention: scalar in
    [0, 1], assumed 0=closed / 1=open (Franka native joint-limit
    convention -- see module docstring's confirmation caveat).

    Returns a torch.FloatTensor (OmniGibson's `env.step()` contract, unlike
    RoboCasa/robosuite's numpy convention -- confirmed via
    `b1k_headless_spike.py`'s `th.zeros(robot.action_dim, ...)`).
    """
    import torch as th

    arm_name, gripper_name = _find_arm_and_gripper_controller_names(robot)
    idx_map = robot.controller_action_idx

    action = th.zeros(robot.action_dim, dtype=th.float32)

    arm_delta = np.asarray(arm_delta, dtype=np.float64)
    scaled = np.empty(6, dtype=np.float64)
    scaled[:3] = arm_delta[:3] * arm_pos_scale
    scaled[3:6] = arm_delta[3:6] * arm_rot_scale

    arm_idx = idx_map[arm_name]
    gripper_idx = idx_map[gripper_name]
    action[arm_idx] = th.as_tensor(scaled[: len(arm_idx)], dtype=th.float32)
    action[gripper_idx] = th.full((len(gripper_idx),), float(gripper_value), dtype=th.float32)

    return action


def robosuite_gripper_to_b1k_smooth(raw_gripper: float) -> float:
    """Convert pi0.5's robosuite-convention gripper value (-1=open, +1=close,
    per `run_robocasa_pi05_eval.py`'s own docstring) into B1K's `smooth`-mode
    [0, 1] scalar, ASSUMING 0=closed / 1=open (Franka native joint-limit
    direction -- confirm empirically before trusting any grasp result, see
    module docstring).

    Linear map: robosuite -1 (open) -> b1k 1.0 (open); robosuite +1 (close)
    -> b1k 0.0 (closed).
    """
    raw_gripper = max(-1.0, min(1.0, float(raw_gripper)))
    return (1.0 - raw_gripper) / 2.0
