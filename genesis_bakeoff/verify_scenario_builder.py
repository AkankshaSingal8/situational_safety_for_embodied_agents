"""
Verification run for scenario_builder.py: build the 1a-i scene once, generate
50 init states (matching SafeLIBERO's 50-episode .pruned_init convention),
save + reload them, and sanity-check the round-trip.
"""
import os

import numpy as np

from scenario_builder import generate_init_states, save_init_states, load_init_states

TABLE_HEIGHT = 0.75
MUG_Z = TABLE_HEIGHT + 0.05
BOTTLE_Z = TABLE_HEIGHT + 0.09

REGION_SPECS = {
    "mug": {"xy_range": ((0.30, -0.06), (0.40, 0.02)), "z": MUG_Z, "footprint_radius": 0.035},
    "bottle": {"xy_range": ((0.30, 0.00), (0.40, 0.16)), "z": BOTTLE_Z, "footprint_radius": 0.030},
}


def build_scene():
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(pos=(0.35, 0.05, TABLE_HEIGHT / 2), size=(0.6, 0.6, TABLE_HEIGHT), fixed=True))
    scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0.0, 0.0, TABLE_HEIGHT)))
    mug = scene.add_entity(
        gs.morphs.Cylinder(pos=(0.35, -0.01, MUG_Z), radius=0.035, height=0.10),
        surface=gs.surfaces.Default(color=(0.8, 0.1, 0.1)),
    )
    bottle = scene.add_entity(
        gs.morphs.Cylinder(pos=(0.35, 0.14, BOTTLE_Z), radius=0.03, height=0.18),
        surface=gs.surfaces.Default(color=(0.1, 0.3, 0.8)),
    )
    scene.build()
    return scene, {"mug": mug, "bottle": bottle}


def main():
    output_path = "/tmp/1a_i_init_states.npz"
    scene, entities = build_scene()

    valid_states, stats = generate_init_states(
        scene, entities, REGION_SPECS, n_episodes=50, settle_steps=80, seed=42,
    )

    print(f"Requested 50 episodes, got {len(valid_states)} valid states")
    print(f"Stats: {stats}")

    if len(valid_states) < 50:
        print("VERIFY_RESULT: SHORT — fewer than 50 valid states produced")
        return

    save_init_states(valid_states, output_path)
    reloaded = load_init_states(output_path)

    assert len(reloaded) == 50, f"round-trip count mismatch: {len(reloaded)} != 50"
    for orig, back in zip(valid_states[:3], reloaded[:3]):
        for name in orig:
            assert np.allclose(orig[name], back[name], atol=1e-5), f"round-trip mismatch for {name}"

    mug_positions = np.array([s["mug"][:2] for s in valid_states])
    bottle_positions = np.array([s["bottle"][:2] for s in valid_states])
    print(f"mug xy spread: {mug_positions.min(axis=0)} to {mug_positions.max(axis=0)}")
    print(f"bottle xy spread: {bottle_positions.min(axis=0)} to {bottle_positions.max(axis=0)}")
    print(f"Saved+reloaded {len(reloaded)} states from {output_path}, round-trip OK")
    print("VERIFY_RESULT: PASS")


if __name__ == "__main__":
    main()
