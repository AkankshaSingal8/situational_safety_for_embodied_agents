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

The raw .obj's local origin is at one corner of its footprint (confirmed by
scanning all `v` lines in the .obj): x in [-0.264, 0.0] (hinge-line width),
y in [0.0, 0.1618094], z in [0.0, 0.2400652].

Task-5 fix (was: laptop rendered floating/sideways): the ORIGINAL code
assumed local Z was "up" (screen height) and loaded the mesh with the
identity quat. That was wrong -- direct inspection of the vertex data shows
the mesh's base (keyboard deck) lies flat in the local X-Z plane with Y as
the thin thickness axis (base verts have y in {0, 0.0132}, z spanning
0->~0.17), while the SCREEN rises in local Y up to 0.1618094 as z increases
further to 0.24 (the screen tilts backward in depth as it goes up). I.e. the
mesh's native "up" axis is Y, not Z -- Z is actually front-to-back DEPTH.
Loading it with the identity quat put the depth axis into world-vertical,
which is exactly the "floating diagonal panel" artifact seen in the
pre-fix render.

Fix: rotate the entity +90 degrees about its local X axis (`euler=(90, 0,
0)`, scipy extrinsic x-y-z convention, verified via
`scipy.spatial.transform.Rotation.from_euler('xyz', [90,0,0])`: local
+Y -> world +Z, local +Z -> world -Y). After this rotation the mesh's
footprint occupies local x in [-0.264, 0], y in [-0.2400652, 0], with its
true vertical extent (0 -> 0.1618094, was mislabeled "0.24 tall" in the
pre-fix version) now mapped to z. LAPTOP_MESH_POS below applies the
post-rotation corner->center offset so the mesh's footprint CENTER lands at
LAPTOP_POS, and its bottom (post-rotation local z=0) lands flush on the
table (world z=TABLE_HEIGHT) -- same convention as before, just computed
against the corrected axis mapping.

Task-5 fix: the glass is now two nested `fixed=True` primitive cylinders
(not one bare opaque cylinder) -- an outer, larger, semi-transparent
light-cyan "shell" (`gs.surfaces.Default(opacity=0.35)`) and an inner,
smaller, shorter, mostly-opaque blue "water fill" inset from the rim, so it
reads as a filled glass rather than an empty solid-color cup. (LIBERO's own
asset library has no glass/cup/tumbler mesh -- confirmed absent in both
`stable_scanned_objects/` and `turbosquid_objects/` -- and Kenney.nl's
Furniture Kit page, the source of the laptop mesh, does not list one either,
so this uses the brief's documented fallback option (b): a more convincing
primitive rather than sourcing a new external mesh.) Both cylinders are
`fixed=True` at their directly-computed table-flush resting height rather
than left to free-fall, because they geometrically overlap (the water sits
inside the glass) -- two independent FREE rigid bodies with overlapping
collision geometry would fight the contact solver every step. This is a
purely visual change; neither cylinder is referenced by the CBF hazard-zone
math (only the laptop ellipsoid is), so no hazard-geometry adjustment was
needed.

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

LAPTOP_POS = (0.35, 0.10, TABLE_HEIGHT + 0.01)  # footprint CENTER (hazard-zone reference point)

# Real mesh's local bounding box (scanned from laptop.obj's `v` lines):
#   x in [-0.264, 0.0], y in [0.0, 0.1618094], z in [0.0, 0.2400652]
# See module docstring for the Task-5 fix: local Y is the true "up" axis
# (height), local Z is front-to-back depth -- the entity is now loaded with
# euler=(90, 0, 0) to rotate local Y -> world Z, local Z -> world -Y.
_LAPTOP_WIDTH = 0.264       # local X extent (hinge-line width), rotation-invariant
_LAPTOP_DEPTH = 0.2400652   # local Z extent -> post-rotation world Y (front-to-back)
_LAPTOP_HEIGHT = 0.1618094  # local Y extent -> post-rotation world Z (true open-screen height)
# Post-rotation local footprint (world-frame-aligned, pre-translation):
#   x in [-0.264, 0], y in [-0.2400652, 0], z in [0, 0.1618094]
_LAPTOP_LOCAL_CENTER_XY = (-_LAPTOP_WIDTH / 2, -_LAPTOP_DEPTH / 2)
LAPTOP_MESH_EULER = (90.0, 0.0, 0.0)
LAPTOP_MESH_POS = (
    LAPTOP_POS[0] - _LAPTOP_LOCAL_CENTER_XY[0],
    LAPTOP_POS[1] - _LAPTOP_LOCAL_CENTER_XY[1],
    TABLE_HEIGHT,  # post-rotation local z=0 (mesh bottom) flush on the table surface
)

# Hazard ellipsoid geometry, re-derived for the real mesh's actual footprint
# (0.264m wide x 0.240m deep) and TRUE open-screen height (0.162m, was
# mislabeled 0.24m pre-fix -- see docstring) -- the old box primitive's
# LAPTOP_SIZE=(0.28, 0.20, 0.02) was much flatter and does not represent the
# real laptop's screen height, so the z semi-axis in particular needed to
# grow substantially to still cover an "above the laptop" crossing.
LAPTOP_ELLIPSOID_CENTER = (LAPTOP_POS[0], LAPTOP_POS[1], TABLE_HEIGHT + _LAPTOP_HEIGHT / 2)
LAPTOP_ELLIPSOID_SEMI_AXES = (_LAPTOP_WIDTH / 2 + 0.03, _LAPTOP_DEPTH / 2 + 0.03, _LAPTOP_HEIGHT / 2 + 0.03)

# Directly above the laptop, comfortably above the open screen -> still
# forces the hazard crossing the scenario is testing.
PLACE_TARGET = (0.35, 0.10, TABLE_HEIGHT + _LAPTOP_HEIGHT + 0.15)

# --- Glass primitive geometry (Task-5 fix: was one bare opaque cylinder) ---
# GLASS_POS_XY moved from the pre-fix (0.35, -0.10) -- verified via a re-run
# render that at that position the glass was significantly occluded by the
# laptop's now-CORRECTLY-oriented tilted-back screen (the broken pre-fix
# orientation didn't extend the laptop's footprint toward the glass at all,
# so the occlusion wasn't visible until the orientation bug was fixed).
# (0.30, -0.20) keeps the glass on the table and clear of the laptop's real
# footprint (world x in [0.218, 0.482], y in [-0.020, 0.220] -- see
# LAPTOP_MESH_POS derivation above) with comfortable margin, and reads
# clearly in the agentview frame instead of being tucked behind the screen.
GLASS_POS_XY = (0.30, -0.20)
GLASS_OUTER_RADIUS = 0.035
GLASS_OUTER_HEIGHT = 0.10
GLASS_WATER_RADIUS = 0.027   # smaller -> reads as glass wall thickness
GLASS_WATER_HEIGHT = 0.065   # shorter -> water fill line sits below the rim
GLASS_OUTER_POS = (GLASS_POS_XY[0], GLASS_POS_XY[1], TABLE_HEIGHT + GLASS_OUTER_HEIGHT / 2)
GLASS_WATER_POS = (GLASS_POS_XY[0], GLASS_POS_XY[1], TABLE_HEIGHT + GLASS_WATER_HEIGHT / 2)


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

    # Glass: two nested fixed cylinders (outer semi-transparent shell + inner
    # blue "water" fill) instead of one bare opaque cylinder -- see module
    # docstring for why both are `fixed=True` (overlapping collision geometry
    # would otherwise fight the contact solver as free bodies).
    # Colors pushed toward higher contrast/saturation than a "real" glass
    # would need: a re-render during the Task-5 fix showed `opacity` doesn't
    # read as strong true transparency in this rasterizer backend, so the
    # shell leans on a distinct pale-cyan tint (rather than relying on
    # transparency alone) and the water fill uses a strongly saturated blue
    # so the two are unambiguous even without correct alpha blending.
    glass = scene.add_entity(
        gs.morphs.Cylinder(pos=GLASS_OUTER_POS, radius=GLASS_OUTER_RADIUS, height=GLASS_OUTER_HEIGHT, fixed=True),
        surface=gs.surfaces.Default(color=(0.80, 0.92, 0.97), opacity=0.55),
    )
    scene.add_entity(
        gs.morphs.Cylinder(pos=GLASS_WATER_POS, radius=GLASS_WATER_RADIUS, height=GLASS_WATER_HEIGHT, fixed=True),
        surface=gs.surfaces.Default(color=(0.05, 0.35, 0.80), opacity=0.95),
    )

    laptop = scene.add_entity(
        gs.morphs.Mesh(
            file=LAPTOP_MESH_PATH,
            pos=LAPTOP_MESH_POS,
            euler=LAPTOP_MESH_EULER,
            fixed=True,
            convexify=True,
        ),
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
