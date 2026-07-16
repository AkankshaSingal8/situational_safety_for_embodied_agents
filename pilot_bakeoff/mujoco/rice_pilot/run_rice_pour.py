import os
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import imageio

HERE = os.path.dirname(os.path.abspath(__file__))
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "rice_pour_scene.xml"))
d = mujoco.MjData(m)

grain_body_ids = [i for i in range(m.nbody) if m.body(i).name and "particle" in m.body(i).name.lower()]
if not grain_body_ids:
    # composite particles are usually auto-named like "particle0_0_0" etc; fall back to
    # "everything except floor/source_bowl/target_bowl/world"
    named = {"world", "floor", "source_bowl", "target_bowl"}
    grain_body_ids = [i for i in range(m.nbody) if m.body(i).name not in named]
print(f"Tracking {len(grain_body_ids)} grain bodies: {[m.body(i).name for i in grain_body_ids[:5]]}...")

renderer = mujoco.Renderer(m, height=480, width=640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.1]
cam.distance = 0.7
cam.azimuth = 90
cam.elevation = -25

frames = []

def render_frame():
    renderer.update_scene(d, camera=cam)
    frames.append(renderer.render().copy())

source_qpos_adr = m.joint("source_bowl_joint").qposadr[0]

# Phase 1: settle grains under gravity (grains start slightly above the source bowl per
# the composite `offset`, so this also validates they fall/settle without exploding).
for step in range(500):
    mujoco.mj_step(m, d)
    if step % 50 == 0:
        render_frame()

print("Post-settle grain z-range:", 
      np.array([d.xpos[i][2] for i in grain_body_ids]).min(), "to",
      np.array([d.xpos[i][2] for i in grain_body_ids]).max())

# Phase 2: scripted "pour" — rotate the source bowl ~100 deg about Y over 1.5s so its open
# side tips toward the target bowl.
n_pour_steps = 750
for step in range(n_pour_steps):
    t = step / n_pour_steps
    angle = t * (100 * np.pi / 180)
    # quaternion for rotation about Y axis
    qw, qy = np.cos(angle / 2), np.sin(angle / 2)
    d.qpos[source_qpos_adr + 3: source_qpos_adr + 7] = [qw, 0, qy, 0]
    d.qpos[source_qpos_adr + 2] += 0.0002  # lift slightly while tilting so it clears the target bowl rim
    mujoco.mj_step(m, d)
    if step % 60 == 0:
        render_frame()

# Phase 3: let everything settle post-pour
for step in range(400):
    mujoco.mj_step(m, d)
    if step % 50 == 0:
        render_frame()

imageio.mimsave(os.path.join(HERE, "rice_pour.mp4"), frames, fps=8)
for i, f in enumerate(frames[::4]):
    imageio.imwrite(os.path.join(HERE, f"frame_{i:03d}.png"), f)

# ── Final classification: source / target / floor / other, by grain XY position ──
target_center = np.array([0.15, 0.0])
source_center_final_xy = d.qpos[source_qpos_adr: source_qpos_adr + 2]

in_target, in_source, on_floor, nan_or_far = 0, 0, 0, 0
final_positions = []
for i in grain_body_ids:
    p = d.xpos[i]
    final_positions.append(p.copy())
    if not np.all(np.isfinite(p)) or np.linalg.norm(p[:2]) > 2.0:
        nan_or_far += 1
        continue
    dist_target = np.linalg.norm(p[:2] - target_center)
    dist_source = np.linalg.norm(p[:2] - source_center_final_xy)
    if dist_target < 0.08 and p[2] < 0.15:
        in_target += 1
    elif dist_source < 0.08:
        in_source += 1
    else:
        on_floor += 1

print(f"RESULT in_target={in_target} in_source={in_source} on_floor={on_floor} nan_or_far={nan_or_far} total={len(grain_body_ids)}")
final_positions = np.array(final_positions)
print("Final grain z stats: min=%.4f max=%.4f mean=%.4f" % (
    final_positions[:,2].min(), final_positions[:,2].max(), final_positions[:,2].mean()))
print("PILOT_RESULT: PASS" if nan_or_far == 0 else "PILOT_RESULT: FAIL (NaN or exploded grains)")
