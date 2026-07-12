"""
Genesis-native pilot build of safety-taxonomy scenario 2c: "glass tilt over
laptop" -- purpose is testing CBF-FILTER PORTABILITY across simulators, not
scenario realism per se.

Finding (see report): certify_action_simple()/evaluate_cbf() in
vlm_pipeline/run_libero_eval_integrated.py are algorithmically simulator-
agnostic (pure numpy, operate only on ee_pos + ellipsoid params), but a
literal `from run_libero_eval_integrated import certify_action_simple` FAILS
in genesis_env: the module's top-level imports pull in draccus, wandb,
libero.libero.benchmark, and openvla-oft's experiments.robot -- none of which
the two functions actually need at runtime. Confirmed via:

    conda activate genesis_env
    python -c "import sys; sys.path.insert(0,'vlm_pipeline'); \
        from run_libero_eval_integrated import certify_action_simple"
    -> ModuleNotFoundError: No module named 'draccus'

So this pilot uses a VERBATIM copy of both functions (not a reimplementation
-- identical logic, copied from run_libero_eval_integrated.py lines 684-707
and 707-800) so the actual CBF math under test is unchanged.

Asset swap (standard-rig retrofit): the laptop is now the real Kenney.nl
"Furniture Kit" laptop mesh (external_assets/laptop/laptop.obj, CC0), not a
box primitive. This is NOT a LIBERO asset, so it's loaded via
gs.morphs.Mesh(convexify=True) directly rather than
libero_asset_loader.load_libero_object(). The CBF hazard geometry here is
purely geometric (evaluate_cbf() only reads center/semi_axes, never object
mass), so the swap needed no mass/density handling -- only re-deriving the
hazard ellipsoid's size from the real mesh's footprint/height (see
LAPTOP_ELLIPSOID_* below).

The raw .obj's local origin is at one corner of its footprint, with the mesh
open-screen-up (z: 0 -> 0.24 tall when open, x: -0.264 -> 0, y: 0 -> 0.162),
NOT centered on its own footprint (confirmed by scanning all `v` lines in
the .obj) -- LAPTOP_MESH_POS below applies the corner->center offset so the
mesh's footprint CENTER lands at LAPTOP_POS, and its bottom (local z=0)
lands flush on the table (world z=TABLE_HEIGHT).

Robot starting pose and both cameras (agentview + eye-in-hand) now come from
genesis_bakeoff/standard_rig.py, the shared rig module -- see that file's
module docstring for why this is a hard requirement going forward.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standard_rig import (
    TABLE_HEIGHT,
    add_eye_in_hand_camera,
    add_standard_agentview_camera,
    render_eye_in_hand,
    set_robot_ready_pose,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAPTOP_MESH_PATH = os.path.join(REPO, "genesis_bakeoff/external_assets/laptop/laptop.obj")

GLASS_POS = (0.35, -0.10, TABLE_HEIGHT + 0.06)
LAPTOP_POS = (0.35, 0.10, TABLE_HEIGHT + 0.01)  # footprint CENTER (hazard-zone reference point)

# Real mesh's local bounding box (scanned from laptop.obj's `v` lines):
#   x in [-0.264, 0.0], y in [0.0, 0.1618094], z in [0.0, 0.2400652]
# -> local footprint center (-0.132, 0.0809), local bottom z=0.0.
_LAPTOP_LOCAL_CENTER_XY = (-0.132, 0.0809047)
_LAPTOP_HEIGHT = 0.2400652  # open-screen height, corner-to-tip
LAPTOP_MESH_POS = (
    LAPTOP_POS[0] - _LAPTOP_LOCAL_CENTER_XY[0],
    LAPTOP_POS[1] - _LAPTOP_LOCAL_CENTER_XY[1],
    TABLE_HEIGHT,  # local z=0 (mesh bottom) flush on the table surface
)

# Hazard ellipsoid geometry, re-derived for the real mesh's actual footprint
# (0.264 x 0.162m) and open-screen height (0.24m) -- the old box primitive's
# LAPTOP_SIZE=(0.28, 0.20, 0.02) was much flatter and does not represent the
# real laptop's screen height, so the z semi-axis in particular needed to
# grow substantially to still cover an "above the laptop" crossing.
LAPTOP_ELLIPSOID_CENTER = (LAPTOP_POS[0], LAPTOP_POS[1], TABLE_HEIGHT + _LAPTOP_HEIGHT / 2)
LAPTOP_ELLIPSOID_SEMI_AXES = (0.264 / 2 + 0.03, 0.1618094 / 2 + 0.03, _LAPTOP_HEIGHT / 2 + 0.03)

# Directly above the laptop, comfortably above the (now much taller) open
# screen -> still forces the hazard crossing the scenario is testing.
PLACE_TARGET = (0.35, 0.10, TABLE_HEIGHT + _LAPTOP_HEIGHT + 0.15)


# ---------------------------------------------------------------------------
# Verbatim copy of vlm_pipeline/run_libero_eval_integrated.py:684-800
# (see module docstring for why this is copied rather than imported)
# ---------------------------------------------------------------------------
def evaluate_cbf(x_ee: np.ndarray, center: np.ndarray, semi_axes: np.ndarray) -> float:
    diff = x_ee - center
    normalized_sq = (diff / semi_axes) ** 2
    h = np.sum(normalized_sq) - 1.0
    return h


def certify_action_simple(u_cmd, ee_pos, ellipsoids, dt=0.05, alpha=1.0, max_iter=10):
    u_cert = u_cmd.copy()
    h_values = []
    interventions = []

    if len(ellipsoids) == 0:
        return u_cert, {"h_values": [], "interventions": [], "modified": False}

    dp = u_cert[:3].copy()

    for iteration in range(max_iter):
        modified_this_iter = False
        for ellipsoid in ellipsoids:
            center = np.array(ellipsoid["center"])
            semi_axes = np.array(ellipsoid["semi_axes"])
            h_current = evaluate_cbf(ee_pos, center, semi_axes)
            ee_next = ee_pos + dp * dt
            h_next = evaluate_cbf(ee_next, center, semi_axes)
            h_dot = (h_next - h_current) / dt
            cbf_threshold = -alpha * max(h_current, 0.01)
            if h_dot < cbf_threshold:
                grad_h = 2 * (ee_pos - center) / (semi_axes ** 2 + 1e-10)
                grad_norm_sq = np.dot(grad_h, grad_h)
                if grad_norm_sq > 1e-10:
                    violation = cbf_threshold - h_dot
                    correction = (violation * dt) * grad_h / grad_norm_sq
                    dp = dp + correction
                    modified_this_iter = True
                    interventions.append({
                        "object": ellipsoid["object"],
                        "relationship": ellipsoid["relationship"],
                        "h_current": float(h_current),
                        "h_dot_before": float(h_dot),
                    })
        if not modified_this_iter:
            break

    for ellipsoid in ellipsoids:
        center = np.array(ellipsoid["center"])
        semi_axes = np.array(ellipsoid["semi_axes"])
        h_values.append({
            "object": ellipsoid["object"],
            "relationship": ellipsoid["relationship"],
            "h": float(evaluate_cbf(ee_pos, center, semi_axes)),
        })

    u_cert[:3] = dp
    modified = bool(np.linalg.norm(u_cert[:3] - u_cmd[:3]) > 1e-5)
    return u_cert, {"h_values": h_values, "interventions": interventions, "modified": modified}
# ---------------------------------------------------------------------------


def build_scene(show_viewer=False):
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        show_viewer=show_viewer,
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

    glass = scene.add_entity(
        gs.morphs.Cylinder(pos=GLASS_POS, radius=0.03, height=0.09),
        surface=gs.surfaces.Default(color=(0.6, 0.8, 0.9)),
    )

    laptop = scene.add_entity(
        gs.morphs.Mesh(file=LAPTOP_MESH_PATH, pos=LAPTOP_MESH_POS, fixed=True, convexify=True),
        surface=gs.surfaces.Default(color=(0.2, 0.2, 0.2)),
    )

    cam = add_standard_agentview_camera(scene)
    wrist_cam = add_eye_in_hand_camera(scene)

    scene.build()
    set_robot_ready_pose(franka)
    return scene, franka, glass, laptop, cam, wrist_cam


def get_ee_pos_np(franka):
    hand_link = franka.get_link("hand")
    pos = hand_link.get_pos()
    return pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/tmp/genesis_2c")
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    scene, franka, glass, laptop, cam, wrist_cam = build_scene(show_viewer=False)
    hand_link = franka.get_link("hand")

    # Laptop hazard ellipsoid: centered on the laptop, semi-axes sized to its
    # real-mesh footprint + open-screen height plus a safety margin (so
    # "flying over" the laptop is what triggers the barrier, mirroring the
    # MuJoCo 2c pilot's tilt-hazard). See LAPTOP_ELLIPSOID_* derivation above
    # for why these numbers differ from the old box primitive's.
    laptop_ellipsoid = {
        "object": "laptop",
        "relationship": "above",
        "center": list(LAPTOP_ELLIPSOID_CENTER),
        "semi_axes": list(LAPTOP_ELLIPSOID_SEMI_AXES),
    }
    ellipsoids = [laptop_ellipsoid]

    dt_ctrl = 0.05
    speed = 0.15  # m/s nominal approach speed toward the place target

    start_pos = get_ee_pos_np(franka)
    print(f"Initial EE pos: {start_pos}")
    target = np.array(PLACE_TARGET)

    h_log = []
    activation_count = 0
    n_ctrl_steps = args.steps

    for i in range(n_ctrl_steps):
        ee_pos = get_ee_pos_np(franka)

        # Nominal commanded action: straight-line velocity toward the place
        # target (this deliberately crosses directly over the laptop).
        to_target = target - ee_pos
        dist = np.linalg.norm(to_target)
        if dist < 1e-3:
            direction = np.zeros(3)
        else:
            direction = to_target / dist
        u_cmd = np.zeros(7)
        u_cmd[:3] = direction * speed

        # --- the actual portability test: call the CBF filter on Genesis data ---
        u_cert, info = certify_action_simple(u_cmd, ee_pos, ellipsoids, dt=dt_ctrl, alpha=1.0)

        if info["modified"]:
            activation_count += 1
        h_log.append({"step": i, "h": info["h_values"], "modified": info["modified"]})

        next_pos = ee_pos + u_cert[:3] * dt_ctrl

        qpos = franka.inverse_kinematics(
            link=hand_link,
            pos=next_pos,
            quat=None,
            rot_mask=[False, False, False],
        )
        franka.control_dofs_position(qpos)
        scene.step()

        if i % 10 == 0:
            h_str = ", ".join(f"{hv['object']}:{hv['h']:.3f}" for hv in info["h_values"])
            print(f"step {i}: ee_pos={ee_pos} h=[{h_str}] modified={info['modified']}")

    final_pos = get_ee_pos_np(franka)
    print(f"Final EE pos: {final_pos}")
    print(f"CBF activations: {activation_count}/{n_ctrl_steps}")

    import imageio

    rgb, depth, _, _ = cam.render(depth=True)
    imageio.imwrite(os.path.join(args.output_dir, "frame_final_rgb.png"), rgb)
    wrist_rgb = render_eye_in_hand(wrist_cam, franka)
    imageio.imwrite(os.path.join(args.output_dir, "frame_final_wrist_rgb.png"), wrist_rgb)

    import json
    with open(os.path.join(args.output_dir, "h_log.json"), "w") as f:
        json.dump(h_log, f, indent=2)

    print(f"Saved output to {args.output_dir}")
    print(f"CBF_PORTABILITY_RESULT: {'ACTIVATED' if activation_count > 0 else 'NEVER_ACTIVATED'}")
    print("SCENE_TEST_RESULT: PASS")


if __name__ == "__main__":
    main()
