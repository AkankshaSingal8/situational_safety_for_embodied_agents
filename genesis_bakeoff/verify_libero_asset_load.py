"""
Step 0 of the scenario build order: verify LIBERO's MJCF object assets load
correctly into Genesis via gs.morphs.MJCF() -- visual mesh + texture +
box-primitive collision geoms all coming through, not just a bare shape.
"""
import os

LIBERO_ASSETS = "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/SafeLIBERO/safelibero/libero/libero/assets"
BOWL_XML = os.path.join(LIBERO_ASSETS, "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml")
MUG_XML = os.path.join(LIBERO_ASSETS, "turbosquid_objects/porcelain_mug/porcelain_mug.xml")


def main():
    import genesis as gs
    import numpy as np
    from libero_asset_loader import load_libero_object

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(pos=(0.0, 0.0, -0.05), size=(0.6, 0.6, 0.02), fixed=True))

    # Drop from higher up (0.25 not 0.05) so a real fall/settle is unambiguous
    # in the position delta, not just sub-mm jitter.
    print(f"Loading bowl from: {BOWL_XML}")
    bowl = load_libero_object(scene, BOWL_XML, pos=(-0.15, 0.0, 0.25))
    print("Bowl loaded OK, entity:", bowl)

    print(f"Loading mug from: {MUG_XML}")
    mug = load_libero_object(scene, MUG_XML, pos=(0.15, 0.0, 0.25))
    print("Mug loaded OK, entity:", mug)

    cam = scene.add_camera(res=(512, 512), pos=(0.0, -0.6, 0.35), lookat=(0.0, 0.0, 0.05), fov=45, GUI=False)

    print("Building scene...")
    scene.build()
    print("Scene built OK.")

    print("Stepping physics 100 steps (settle under gravity)...")
    for _ in range(100):
        scene.step()

    def to_np(x):
        return np.array(x.cpu() if hasattr(x, "cpu") else x)

    bowl_pos = to_np(bowl.get_pos())
    mug_pos = to_np(mug.get_pos())
    print(f"Bowl final pos: {bowl_pos}")
    print(f"Mug final pos: {mug_pos}")

    rgb, depth, _, _ = cam.render(depth=True)
    out_dir = "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/genesis_bakeoff/libero_asset_verify_output"
    os.makedirs(out_dir, exist_ok=True)
    import imageio
    imageio.imwrite(os.path.join(out_dir, "bowl_mug_rgb.png"), rgb)
    print(f"Saved render to {out_dir}/bowl_mug_rgb.png")

    finite = np.all(np.isfinite(bowl_pos)) and np.all(np.isfinite(mug_pos))
    reasonable_height = bowl_pos[2] > -0.1 and mug_pos[2] > -0.1
    # Started at z=0.25 -- a real free-joint fall/settle under gravity must
    # have moved meaningfully (not stayed pinned at 0.25, and not tunneled
    # through the floor to some large negative value).
    bowl_fell = bowl_pos[2] < 0.20
    mug_fell = mug_pos[2] < 0.20
    print(f"bowl_fell={bowl_fell} (moved from 0.25 to {bowl_pos[2]:.4f})")
    print(f"mug_fell={mug_fell} (moved from 0.25 to {mug_pos[2]:.4f})")
    ok = finite and reasonable_height and bowl_fell and mug_fell
    print("LIBERO_ASSET_LOAD_RESULT: PASS" if ok else "LIBERO_ASSET_LOAD_RESULT: FAIL")


if __name__ == "__main__":
    main()
