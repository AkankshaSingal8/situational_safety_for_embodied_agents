"""
Spike: validate a generic placement_sampler.py against the 1a-i scene.
Builds the scene ONCE, then repositions mug/bottle via set_pos for each of
N sampled trials, settles physics, and checks for interpenetration/explosion.
"""
import argparse
import os

import numpy as np

from placement_sampler import sample_placement

TABLE_HEIGHT = 0.75
MUG_Z = TABLE_HEIGHT + 0.05
BOTTLE_Z = TABLE_HEIGHT + 0.09

# Widened vs. the BDDL pilot's literal 2cm ranges (0.00-0.02) so rejection
# sampling is actually exercised -- the original BDDL ranges are too tight
# for meaningful overlap-rejection testing (any two 2cm-box samples rarely
# collide given 3.5cm/3cm object radii). Same origin/scale, wider spread.
REGION_SPECS = {
    "mug": {
        "xy_range": ((0.30, -0.06), (0.40, 0.02)),
        "z": MUG_Z,
        "footprint_radius": 0.035,
    },
    "bottle": {
        "xy_range": ((0.30, 0.00), (0.40, 0.16)),
        "z": BOTTLE_Z,
        "footprint_radius": 0.030,
    },
}


def build_scene():
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(pos=(0.35, 0.05, TABLE_HEIGHT / 2), size=(0.6, 0.6, TABLE_HEIGHT), fixed=True))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0.0, 0.0, TABLE_HEIGHT)))
    mug = scene.add_entity(
        gs.morphs.Cylinder(pos=(0.35, -0.01, MUG_Z), radius=0.035, height=0.10),
        surface=gs.surfaces.Default(color=(0.8, 0.1, 0.1)),
    )
    bottle = scene.add_entity(
        gs.morphs.Cylinder(pos=(0.35, 0.14, BOTTLE_Z), radius=0.03, height=0.18),
        surface=gs.surfaces.Default(color=(0.1, 0.3, 0.8)),
    )
    cam = scene.add_camera(res=(512, 512), pos=(1.4, 0.3, TABLE_HEIGHT + 0.9),
                            lookat=(0.35, 0.05, TABLE_HEIGHT + 0.05), fov=45, GUI=False)
    scene.build()
    return scene, mug, bottle, cam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/tmp/genesis_placement_spike")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--settle_steps", type=int, default=80)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    scene, mug, bottle, cam = build_scene()

    print(f"has set_pos: mug={hasattr(mug, 'set_pos')} bottle={hasattr(bottle, 'set_pos')}")

    first_try_success = 0
    retry_success = 0
    total_geometric_fail = 0
    physics_bad = 0
    valid_trials = []

    for trial in range(args.n_trials):
        placement = sample_placement(REGION_SPECS, max_tries=200, seed=1000 + trial)
        if placement is None:
            total_geometric_fail += 1
            print(f"trial {trial}: GEOMETRIC SAMPLING FAILED (exhausted max_tries)")
            continue

        mug_xy = placement["mug"][:2]
        bottle_xy = placement["bottle"][:2]
        dist = np.hypot(mug_xy[0] - bottle_xy[0], mug_xy[1] - bottle_xy[1])
        if dist >= 0.065:  # sum of radii, i.e. would have passed on a truly-first geometric draw region-wide
            first_try_success += 1
        else:
            retry_success += 1  # sampler internally retried to reach a valid config

        mug.set_pos(np.array(placement["mug"]))
        bottle.set_pos(np.array(placement["bottle"]))
        mug.set_dofs_velocity(np.zeros(6)) if hasattr(mug, "set_dofs_velocity") else None
        bottle.set_dofs_velocity(np.zeros(6)) if hasattr(bottle, "set_dofs_velocity") else None

        for _ in range(args.settle_steps):
            scene.step()

        mug_final = np.array(mug.get_pos().cpu() if hasattr(mug.get_pos(), "cpu") else mug.get_pos())
        bottle_final = np.array(bottle.get_pos().cpu() if hasattr(bottle.get_pos(), "cpu") else bottle.get_pos())
        final_dist = np.hypot(mug_final[0] - bottle_final[0], mug_final[1] - bottle_final[1])

        bad = False
        if not np.all(np.isfinite(mug_final)) or not np.all(np.isfinite(bottle_final)):
            bad = True
        if abs(mug_final[2] - MUG_Z) > 0.08 or abs(bottle_final[2] - BOTTLE_Z) > 0.08:
            bad = True  # fell through table / launched
        if final_dist < 0.045:  # objects ended up interpenetrating despite geometric OK
            bad = True
        if bad:
            physics_bad += 1

        print(f"trial {trial}: mug_xy={placement['mug'][:2]} bottle_xy={placement['bottle'][:2]} "
              f"init_dist={dist:.3f} final_dist={final_dist:.3f} mug_z={mug_final[2]:.3f} "
              f"bottle_z={bottle_final[2]:.3f} {'BAD' if bad else 'OK'}")

        valid_trials.append(trial)

        if trial in (0, 5, 10, 15) and len(valid_trials) <= 4:
            rgb, _, _, _ = cam.render(depth=False)
            import imageio
            imageio.imwrite(os.path.join(args.output_dir, f"trial_{trial:02d}.png"), rgb)

    print(f"\nSummary: {args.n_trials} trials requested")
    print(f"  geometric_fail={total_geometric_fail}")
    print(f"  first_try_geometric_ok={first_try_success}")
    print(f"  retry_needed_geometric_ok={retry_success}")
    print(f"  physics_settling_revealed_bad={physics_bad}")
    print(f"  saved {min(4, len(valid_trials))} sample frames to {args.output_dir}")
    print("PLACEMENT_SPIKE_RESULT: PASS" if physics_bad == 0 and total_geometric_fail == 0 else "PLACEMENT_SPIKE_RESULT: ISSUES_FOUND")


if __name__ == "__main__":
    main()
