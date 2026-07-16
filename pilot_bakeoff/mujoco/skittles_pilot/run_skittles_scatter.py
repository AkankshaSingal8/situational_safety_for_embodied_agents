import os
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import imageio

HERE = os.path.dirname(os.path.abspath(__file__))
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "skittles_scatter_scene.xml"))
d = mujoco.MjData(m)

named = {"world", "floor", "bag", "target_bowl"}
candy_body_ids = [i for i in range(m.nbody) if m.body(i).name not in named]
print(f"Tracking {len(candy_body_ids)} candy bodies: {[m.body(i).name for i in candy_body_ids[:5]]}...")

renderer = mujoco.Renderer(m, height=480, width=640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.1]
cam.distance = 0.75
cam.azimuth = 90
cam.elevation = -25

frames = []

def render_frame():
    renderer.update_scene(d, camera=cam)
    frames.append(renderer.render().copy())

bag_qpos_adr = m.joint("bag_joint").qposadr[0]

# Phase 1: settle candies under gravity in the bag.
for step in range(500):
    mujoco.mj_step(m, d)
    if step % 50 == 0:
        render_frame()

print("Post-settle candy z-range:",
      np.array([d.xpos[i][2] for i in candy_body_ids]).min(), "to",
      np.array([d.xpos[i][2] for i in candy_body_ids]).max())

# Phase 2: abrupt bag-tip release — unlike the rice pilot's gentle 100deg/1.5s pour, this
# rotates the bag ~140deg in 0.4s (nearly 4x the angular rate) to simulate an accidental
# tip/tear rather than a controlled pour, producing higher release velocity and a wider,
# more chaotic scatter.
n_tip_steps = 200
for step in range(n_tip_steps):
    t = step / n_tip_steps
    angle = t * (140 * np.pi / 180)
    qw, qy = np.cos(angle / 2), np.sin(angle / 2)
    d.qpos[bag_qpos_adr + 3: bag_qpos_adr + 7] = [qw, 0, qy, 0]
    d.qpos[bag_qpos_adr + 2] += 0.0015  # lift while tipping, faster than the rice pilot's 0.0002/step
    mujoco.mj_step(m, d)
    if step % 15 == 0:
        render_frame()

# Phase 3: let everything settle/bounce out post-tip.
for step in range(500):
    mujoco.mj_step(m, d)
    if step % 50 == 0:
        render_frame()

imageio.mimsave(os.path.join(HERE, "skittles_scatter.mp4"), frames, fps=8)
for i, f in enumerate(frames[::4]):
    imageio.imwrite(os.path.join(HERE, f"frame_{i:03d}.png"), f)

# ── Final classification: target / bag / floor / other, by candy XY position ──
target_center = np.array([0.15, 0.0])
bag_center_final_xy = d.qpos[bag_qpos_adr: bag_qpos_adr + 2]

in_target, in_bag, on_floor, nan_or_far = 0, 0, 0, 0
final_positions = []
xy_spread_pts = []
for i in candy_body_ids:
    p = d.xpos[i]
    final_positions.append(p.copy())
    if not np.all(np.isfinite(p)) or np.linalg.norm(p[:2]) > 2.0:
        nan_or_far += 1
        continue
    xy_spread_pts.append(p[:2].copy())
    dist_target = np.linalg.norm(p[:2] - target_center)
    dist_bag = np.linalg.norm(p[:2] - bag_center_final_xy)
    if dist_target < 0.08 and p[2] < 0.12:
        in_target += 1
    elif dist_bag < 0.07:
        in_bag += 1
    else:
        on_floor += 1

print(f"RESULT in_target={in_target} in_bag={in_bag} on_floor={on_floor} nan_or_far={nan_or_far} total={len(candy_body_ids)}")
final_positions = np.array(final_positions)
print("Final candy z stats: min=%.4f max=%.4f mean=%.4f" % (
    final_positions[:,2].min(), final_positions[:,2].max(), final_positions[:,2].mean()))

if xy_spread_pts:
    xy_spread_pts = np.array(xy_spread_pts)
    spread_x = xy_spread_pts[:,0].max() - xy_spread_pts[:,0].min()
    spread_y = xy_spread_pts[:,1].max() - xy_spread_pts[:,1].min()
    print("Final XY spread: %.3fm x %.3fm (rice pilot reference: contained pour, spread << 0.4m)" % (spread_x, spread_y))

print("PILOT_RESULT: PASS" if nan_or_far == 0 else "PILOT_RESULT: FAIL (NaN or exploded candies)")
