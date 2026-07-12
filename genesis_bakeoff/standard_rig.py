"""
Standard robot pose + camera rig, shared across ALL Genesis bake-off
scenarios.

*** THIS MODULE IS A HARD REQUIREMENT GOING FORWARD ***
Every scenario script built from here on (starting with 4a) MUST import and
use `set_robot_ready_pose`, `add_standard_agentview_camera`, and (whenever
the scenario has a Franka arm) `render_eye_in_hand` / `add_eye_in_hand_camera`
from this module instead of hand-rolling its own camera pose or robot-init
qpos. Do not redefine TABLE_HEIGHT, camera pos/fov, or the ready-pose qpos
locally in a new scenario file -- import them from here so all scenarios stay
visually and quantitatively comparable (mirrors LIBERO's own convention of a
single fixed camera rig across all its benchmark tasks).

Background: every scenario script built before this module existed
(scenario_1a_i.py, scenario_1b.py, scenario_2c.py, scenario_3a_i.py) picked
its own agentview camera pos/lookat ad hoc and relied on Genesis's raw MJCF
default init for the robot's starting joint configuration -- which Genesis's
own scene-build logs warn is "Neutral robot position (qpos0) exceeds joint
limits." This module standardizes both, plus adds a wrist ("eye-in-hand")
camera capability that didn't exist anywhere in this project before.

Genesis has no native "attach camera to a moving link" API (confirmed during
the OpenVLA-Genesis spike, see spike_openvla_e2e.py's update_wrist_camera for
the original workaround this module generalizes): a free camera's pose must
be manually recomputed from the hand link's world pos/quat every frame,
immediately before each render call.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Matches the de facto convention already used (redundantly) in every
# existing scenario script -- typical SafeLIBERO/robosuite table-top height.
TABLE_HEIGHT = 0.75

# Standard Franka Panda "ready" pose, as used throughout the robosuite/LIBERO
# ecosystem (robosuite's `robots/manipulators/panda_robot.py`
# init_qpos). 7 arm joints + gripper fully open.
ARM_READY_QPOS = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398])
GRIPPER_OPEN = np.array([0.04, 0.04])  # finger joints, Genesis's bundled panda.xml convention
GRIPPER_CLOSED = np.array([0.0, 0.0])

# Standard Genesis panda.xml dof layout: 7 arm dofs followed by 2 finger dofs
# (confirmed in scenario_1b.py / spike_openvla_e2e.py -- "Panda: 7 arm joints
# + 2 finger joints = 9 dofs").
ARM_DOFS_IDX = list(range(7))
GRIPPER_DOFS_IDX = [7, 8]

# Standard agentview camera parameters. Taken verbatim from scenario_1a_i.py
# (the first scenario built in this project, and the one closest to this
# project's documented WORKSPACE_BOUNDS convention in the top-level
# CLAUDE.md) -- do NOT change pos/fov/res here without re-verifying every
# scenario's framing, since this is now the shared standard.
AGENTVIEW_RES = (512, 512)
AGENTVIEW_POS = (1.4, 0.3, TABLE_HEIGHT + 0.9)
AGENTVIEW_FOV = 45
DEFAULT_TABLE_CENTER_XY = (0.35, 0.05)

# Eye-in-hand (wrist) camera parameters. Kept at the SAME res/fov as
# agentview for cross-scenario/cross-view comparability, per this task's
# brief -- a real wrist camera often has a narrower FOV, but consistency
# with the agentview convention was judged more valuable for this benchmark
# than photorealism of the FOV number itself. Document any future deviation
# here if a scenario has a specific reason to need a narrower/wider wrist FOV.
EYE_IN_HAND_RES = (512, 512)
EYE_IN_HAND_FOV = 45
# How far "in front of" the gripper (along its approach axis) the wrist
# camera's lookat target is placed -- matches the empirically-verified 0.15m
# standoff used in spike_openvla_e2e.py's update_wrist_camera.
EYE_IN_HAND_LOOKAHEAD = 0.15


def set_robot_ready_pose(franka, arm_dofs_idx=None, gripper_dofs_idx=None, gripper_open=True):
    """
    Sets `franka` directly to the standard ready pose (7 arm joints @
    ARM_READY_QPOS, gripper open by default), bypassing Genesis's raw MJCF
    default init (which Genesis's own scene-build logs warn exceeds joint
    limits). Uses `set_dofs_position` on the full 9-dof vector at once --
    the same API already used for per-episode resets in scenario_1b.py.

    Call this once, right after `scene.build()`, in every scenario's
    build_scene() (and again on every episode reset, if the scenario resets
    the robot between episodes -- see scenario_1b.py's run_episode()).
    """
    if arm_dofs_idx is None:
        arm_dofs_idx = ARM_DOFS_IDX
    if gripper_dofs_idx is None:
        gripper_dofs_idx = GRIPPER_DOFS_IDX
    gripper_q = GRIPPER_OPEN if gripper_open else GRIPPER_CLOSED
    full_dofs_idx = list(arm_dofs_idx) + list(gripper_dofs_idx)
    full_q = np.concatenate([ARM_READY_QPOS, gripper_q])
    franka.set_dofs_position(full_q, full_dofs_idx)


def add_standard_agentview_camera(scene, table_center_xy=DEFAULT_TABLE_CENTER_XY):
    """
    Returns a camera built with FIXED, standardized agentview parameters:
    res=(512, 512), pos=(1.4, 0.3, TABLE_HEIGHT + 0.9), fov=45, GUI=False.

    Only `lookat`'s target point may vary per scenario (via table_center_xy,
    defaulting to the single shared standard center (0.35, 0.05)) -- pos/fov/
    res are NOT parameterized here on purpose, so every scenario's agentview
    frame is directly visually comparable. If a scenario ever needs a
    genuinely different lookat center, document why in that scenario's own
    code comments (do not silently drift from the shared default).
    """
    cx, cy = table_center_xy
    return scene.add_camera(
        res=AGENTVIEW_RES,
        pos=AGENTVIEW_POS,
        lookat=(cx, cy, TABLE_HEIGHT + 0.05),
        fov=AGENTVIEW_FOV,
        GUI=False,
    )


def add_eye_in_hand_camera(scene):
    """
    Adds a free camera to be used as the eye-in-hand ("wrist") view. Its
    initial pos/lookat are placeholders -- `render_eye_in_hand()` must be
    called (which repositions it from the hand link's live pose) before
    every render, since Genesis has no native camera-to-link attachment.
    """
    return scene.add_camera(
        res=EYE_IN_HAND_RES,
        pos=(0.0, 0.0, 1.0),
        lookat=(0.0, 0.0, 0.0),
        fov=EYE_IN_HAND_FOV,
        GUI=False,
    )


def _to_numpy(x):
    """Safe tensor/array -> numpy conversion. Genesis entity/link getters can
    return either plain arrays or CUDA torch tensors depending on call site
    (confirmed empirically: hand.get_pos()/.get_quat() convert fine via bare
    np.asarray(), but franka.get_dofs_position() returned a live cuda:0
    tensor that raised `TypeError: can't convert cuda:0 device type tensor
    to numpy` -- same guarded pattern already used for this reason in
    scenario_1b.py's get_arm_q_np/get_xy_np)."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _quat_wxyz_to_rotmat(quat_wxyz):
    """Genesis (MJCF-loaded, MuJoCo-convention) quats are (w, x, y, z).
    Same rotation-matrix construction as spike_openvla_e2e.py's
    update_wrist_camera, factored out here for reuse."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def render_eye_in_hand(cam, franka, hand_link_name="hand", offset_local=(0.0, 0.0, 0.05)):
    """
    Repositions `cam` to a wrist-mounted vantage derived from
    `franka.get_link(hand_link_name)`'s live world pos/quat, then renders and
    returns the RGB frame. No native "attach camera to link" API exists in
    this Genesis version (confirmed during the OpenVLA-Genesis spike), so
    this must be called immediately before every render, every frame/step.

    offset_local=(lateral_x, lateral_y, forward): a small standoff offset
    expressed in the hand's own local frame, mimicking a real wrist-mounted
    camera that sits slightly off the gripper's exact origin. The Panda
    hand link's approach axis (the direction the gripper points) is local
    -z (verified visually in spike_openvla_e2e.py's saved wrist frames);
    `forward` (the offset_local z-component) is interpreted as a standoff
    distance measured FORWARD along that approach axis -- i.e.
    offset_local=(0, 0, 0.05) means "5cm in front of the gripper," not "5cm
    along the raw local +z axis."
    """
    hand = franka.get_link(hand_link_name)
    hand_pos = _to_numpy(hand.get_pos()).flatten()
    hand_quat_wxyz = _to_numpy(hand.get_quat()).flatten()
    R = _quat_wxyz_to_rotmat(hand_quat_wxyz)

    approach_local = np.array([0.0, 0.0, -1.0])
    lateral_local = np.array([offset_local[0], offset_local[1], 0.0])
    cam_pos = hand_pos + R @ (lateral_local + approach_local * offset_local[2])
    lookat = cam_pos + (R @ approach_local) * EYE_IN_HAND_LOOKAHEAD

    cam.set_pose(pos=tuple(cam_pos), lookat=tuple(lookat))
    rgb, _depth, _seg, _normal = cam.render(depth=True)
    return rgb


if __name__ == "__main__":
    # Minimal standalone smoke test: build a bare scene with just a Franka,
    # apply the ready pose, add both cameras, and render one frame from each.
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0.0, 0.0, TABLE_HEIGHT)))
    agentview_cam = add_standard_agentview_camera(scene)
    wrist_cam = add_eye_in_hand_camera(scene)
    scene.build()

    set_robot_ready_pose(franka)
    q_immediate = _to_numpy(franka.get_dofs_position(ARM_DOFS_IDX)).flatten()
    assert np.allclose(q_immediate, ARM_READY_QPOS, atol=1e-4), (q_immediate, ARM_READY_QPOS)
    print(f"Ready-pose sanity OK immediately after set_robot_ready_pose: arm_q={q_immediate}")

    # NOTE: set_dofs_position() teleports state instantly but does not itself
    # command the actuators to HOLD there -- a few free scene.step() calls
    # with no active position controller let gravity drift the arm slightly
    # off the exact ready qpos (confirmed empirically: ~0.01-0.04 rad drift
    # over 5 steps). That is expected physical behavior, not a bug in
    # set_robot_ready_pose -- scenario scripts that need the pose to persist
    # issue their own control_dofs_position calls (see scenario_1b.py's
    # move_to()). Use a looser tolerance here accordingly.
    for _ in range(5):
        scene.step()
    q = _to_numpy(franka.get_dofs_position(ARM_DOFS_IDX)).flatten()
    assert np.allclose(q, ARM_READY_QPOS, atol=0.1), (q, ARM_READY_QPOS)
    print(f"Ready-pose sanity OK after 5 free steps (gravity drift expected): arm_q={q}")

    agent_rgb, _, _, _ = agentview_cam.render()
    wrist_rgb = render_eye_in_hand(wrist_cam, franka)
    print(f"agentview frame shape={agent_rgb.shape}, wrist frame shape={wrist_rgb.shape}")
    print("STANDARD_RIG_SMOKE_TEST: PASS")
