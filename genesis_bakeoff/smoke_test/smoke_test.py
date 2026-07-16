import sys
import numpy as np

print("--- Importing genesis ---")
import genesis as gs

print("--- Initializing genesis (gpu backend) ---")
gs.init(backend=gs.gpu, logging_level="warning")

print("--- Building minimal rigid-body scene ---")
scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(dt=0.01),
)
plane = scene.add_entity(gs.morphs.Plane())
box = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0, 0, 0.5)))

cam = scene.add_camera(
    res=(256, 256),
    pos=(1.5, 0.0, 1.0),
    lookat=(0.0, 0.0, 0.3),
    fov=45,
    GUI=False,
)

print("--- Building scene (this triggers GPU kernel compilation) ---")
scene.build()

print("--- Stepping physics 50 steps ---")
for i in range(50):
    scene.step()

print("--- Rendering headless RGB + depth frame ---")
rgb, depth, _, _ = cam.render(depth=True)
print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")

final_box_pos = box.get_pos()
print(f"Box final position after 50 steps (should have fallen from z=0.5): {final_box_pos}")

assert rgb.shape[:2] == (256, 256), "RGB render shape mismatch"
assert depth.shape[:2] == (256, 256), "Depth render shape mismatch"
assert float(final_box_pos[2]) < 0.5, "Box did not fall — physics step may not be working"

print("SMOKE_TEST_RESULT: PASS")
