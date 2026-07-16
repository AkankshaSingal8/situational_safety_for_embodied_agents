"""
Spike test: does OpenVLA-OFT run end-to-end against a Genesis scene?

Builds a Genesis scene (Franka Panda + mug + bottle, same layout as
scenario_1a_i.py), renders agentview + wrist camera observations, feeds them
through the UNMODIFIED OpenVLA-OFT get_action() call (imported from the
existing MuJoCo pipeline), converts the returned 7-dim delta-pose action into
a Genesis IK-driven joint target, and steps the sim. Logs EE position and
gripper state every step so motion sanity can be checked after the run.

No existing Genesis-VLA adapter exists anywhere to copy from — this script
IS that adapter, written from scratch for this spike.
"""
import argparse
import os
import sys

import numpy as np

TABLE_HEIGHT = 0.75
MUG_POS = (0.35, -0.01, TABLE_HEIGHT + 0.05)
BOTTLE_POS = (0.35, 0.14, TABLE_HEIGHT + 0.09)

REPO = "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents"


def quat2axisangle(quat):
    """Copied verbatim from openvla-oft/experiments/robot/libero/libero_utils.py
    (itself copied from robosuite). Expects (x,y,z,w) convention."""
    import math
    quat = np.array(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def wxyz_to_xyzw(q):
    """Genesis (MJCF-loaded, MuJoCo-convention) quats are (w,x,y,z).
    robosuite/OpenVLA-OFT expect (x,y,z,w)."""
    q = np.asarray(q, dtype=np.float64).flatten()
    return np.array([q[1], q[2], q[3], q[0]])


def build_scene():
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=0.01),
    )

    scene.add_entity(gs.morphs.Plane())

    scene.add_entity(
        gs.morphs.Box(
            pos=(0.35, 0.05, TABLE_HEIGHT / 2),
            size=(0.6, 0.6, TABLE_HEIGHT),
            fixed=True,
        ),
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.0, 0.0, TABLE_HEIGHT),
        ),
    )

    mug = scene.add_entity(
        gs.morphs.Cylinder(pos=MUG_POS, radius=0.035, height=0.10),
        surface=gs.surfaces.Default(color=(0.8, 0.1, 0.1)),
    )

    bottle = scene.add_entity(
        gs.morphs.Cylinder(pos=BOTTLE_POS, radius=0.03, height=0.18),
        surface=gs.surfaces.Default(color=(0.1, 0.3, 0.8)),
    )

    # agentview: fixed external camera
    agentview_cam = scene.add_camera(
        res=(256, 256),
        pos=(1.4, 0.3, TABLE_HEIGHT + 0.9),
        lookat=(0.35, 0.05, TABLE_HEIGHT + 0.05),
        fov=45,
        GUI=False,
    )

    # wrist camera: no native link-attachment API found in this Genesis version's
    # RigidEntity/Camera surface (no attach_to_link-style method on gs.Camera) —
    # manually re-pose a free camera to the hand link's pose each step instead.
    wrist_cam = scene.add_camera(
        res=(256, 256),
        pos=(0.0, 0.0, 1.0),
        lookat=(0.0, 0.0, 0.0),
        fov=60,
        GUI=False,
    )

    scene.build()

    return scene, franka, mug, bottle, agentview_cam, wrist_cam


def update_wrist_camera(wrist_cam, franka):
    """Re-pose the wrist camera to follow the hand link each step (no native
    link-attach API found)."""
    hand = franka.get_link("hand")
    hand_pos = np.asarray(hand.get_pos()).flatten()
    # Look outward along the gripper's approach axis (approximate: -z of hand frame
    # is commonly the approach direction for Panda; verified visually via saved
    # frames rather than assumed blindly).
    hand_quat_wxyz = np.asarray(hand.get_quat()).flatten()
    # Rotate a forward vector (0,0,-1) by the hand quaternion (wxyz) manually.
    w, x, y, z = hand_quat_wxyz
    fwd = np.array([0.0, 0.0, -1.0])
    # quaternion rotation: v' = q v q^-1, implemented via rotation matrix for clarity
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    fwd_world = R @ fwd
    lookat = hand_pos + fwd_world * 0.15
    wrist_cam.set_pose(pos=tuple(hand_pos), lookat=tuple(lookat))


def get_gripper_qpos(franka, gripper_dofs_idx):
    return np.asarray(franka.get_dofs_position(gripper_dofs_idx)).flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", default="moojink/openvla-7b-oft-finetuned-libero-spatial")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--output_dir", default="/tmp/genesis_openvla_spike")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sys.path.insert(0, os.path.join(REPO, "vlm_pipeline"))
    sys.path.insert(0, os.path.join(REPO, "openvla-oft"))

    print("--- Loading OpenVLA-OFT model ---")
    from run_libero_eval_integrated import GenerateConfig, initialize_model, process_action
    from experiments.robot.robot_utils import get_action as openvla_get_action

    cfg = GenerateConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        task_suite_name="safelibero_spatial",
        env_img_res=256,
    )
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    print("--- Model loaded OK ---")

    print("--- Building Genesis scene ---")
    scene, franka, mug, bottle, agentview_cam, wrist_cam = build_scene()

    n_dofs = franka.n_dofs
    print(f"Franka n_dofs: {n_dofs}")
    # Panda: 7 arm joints + 2 finger joints = 9 dofs (standard Genesis panda.xml)
    arm_dofs_idx = list(range(7))
    gripper_dofs_idx = list(range(7, 9))
    hand_link = franka.get_link("hand")

    resize_size = 224
    import imageio

    ee_log = []
    gripper_log = []
    error = None

    try:
        for step in range(args.steps):
            update_wrist_camera(wrist_cam, franka)

            agent_rgb, _, _, _ = agentview_cam.render()
            wrist_rgb, _, _, _ = wrist_cam.render()

            ee_pos = np.asarray(hand_link.get_pos()).flatten()
            ee_quat_wxyz = np.asarray(hand_link.get_quat()).flatten()
            ee_quat_xyzw = wxyz_to_xyzw(ee_quat_wxyz)
            gripper_qpos = get_gripper_qpos(franka, gripper_dofs_idx)

            state = np.concatenate((ee_pos, quat2axisangle(ee_quat_xyzw), gripper_qpos))

            from PIL import Image
            img_resized = np.array(Image.fromarray(agent_rgb).resize((resize_size, resize_size)))
            wrist_resized = np.array(Image.fromarray(wrist_rgb).resize((resize_size, resize_size)))

            observation = {
                "full_image": img_resized,
                "wrist_image": wrist_resized,
                "state": state,
            }

            action = openvla_get_action(
                cfg, model, observation, "pick up the mug", processor=processor,
                action_head=action_head, proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector, use_film=cfg.use_film,
            )
            if isinstance(action, list):
                action = action[0]
            action = process_action(np.asarray(action, dtype=np.float64), "openvla")

            delta_pos = action[:3] * 0.05  # scale delta similar to robosuite OSC step size
            target_pos = ee_pos + delta_pos
            target_quat_xyzw = ee_quat_xyzw  # keep orientation fixed for this spike (rotation deltas skipped for simplicity)
            target_quat_wxyz = np.array([target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]])

            qpos_target = franka.inverse_kinematics(
                link=hand_link, pos=target_pos, quat=target_quat_wxyz,
            )
            franka.control_dofs_position(qpos_target[:7], arm_dofs_idx)

            gripper_cmd = action[6]
            gripper_target = np.array([0.04, 0.04]) if gripper_cmd < 0 else np.array([0.0, 0.0])
            franka.control_dofs_position(gripper_target, gripper_dofs_idx)

            scene.step()

            ee_log.append(target_pos.copy())
            gripper_log.append(float(gripper_cmd))
            print(f"step {step}: ee_pos={ee_pos} action[:3]={action[:3]} gripper_cmd={gripper_cmd:.3f}")

            if step in (0, args.steps // 2, args.steps - 1):
                imageio.imwrite(os.path.join(args.output_dir, f"agentview_step{step}.png"), agent_rgb)
                imageio.imwrite(os.path.join(args.output_dir, f"wrist_step{step}.png"), wrist_rgb)

    except Exception as e:
        import traceback
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print("SPIKE_ERROR:")
        print(error)

    ee_arr = np.array(ee_log) if ee_log else np.zeros((0, 3))
    print(f"\nTotal steps completed: {len(ee_log)}/{args.steps}")
    if len(ee_arr) > 1:
        total_disp = np.linalg.norm(ee_arr[-1] - ee_arr[0])
        max_step_disp = np.max(np.linalg.norm(np.diff(ee_arr, axis=0), axis=1))
        print(f"Total EE displacement: {total_disp:.4f} m")
        print(f"Max single-step EE displacement: {max_step_disp:.4f} m")
        print(f"Any NaN in EE log: {np.isnan(ee_arr).any()}")

    if error is None and len(ee_log) == args.steps:
        print("SPIKE_RESULT: PASS")
    elif len(ee_log) > 0:
        print("SPIKE_RESULT: PARTIAL")
    else:
        print("SPIKE_RESULT: FAIL")


if __name__ == "__main__":
    main()
