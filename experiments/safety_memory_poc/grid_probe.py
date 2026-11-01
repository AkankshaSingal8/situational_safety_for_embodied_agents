"""Coarse reachability map at transit height, so the scored-episode geometry is
chosen from measurement instead of assumption."""
import os, json
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from src.policy import waypoint_action

bm = benchmark.get_benchmark_dict()["safelibero_spatial"]()
task = bm.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = ControlEnv(bddl, use_camera_obs=False, has_offscreen_renderer=False,
                 has_renderer=False, camera_names=[], control_freq=20,
                 hard_reset=False, horizon=100000, ignore_done=True)
env.seed(1000); obs = env.reset()
objs = {k[:-4]: np.asarray(obs[k])[:3] for k in obs
        if k.endswith("_pos") and "robot0" not in k and "to_robot0" not in k}
print("objects:", {k: np.round(v, 3).tolist() for k, v in objs.items()}, flush=True)

def go(target, steps=220):
    env.seed(1000); o = env.reset()
    best = 1e9
    for _ in range(steps):
        p = np.asarray(o["robot0_eef_pos"])[:3]
        best = min(best, float(np.linalg.norm(p - target)))
        o, _, _, _ = env.step(waypoint_action(p, target))
    return best, np.asarray(o["robot0_eef_pos"])[:3]

Z = 1.01
xs = np.round(np.arange(-0.40, 0.281, 0.08), 3)
ys = np.round(np.arange(-0.08, 0.401, 0.08), 3)
grid = {}
print(f"\nreachability at z={Z} (value = best distance achieved, m)", flush=True)
hdr = "   x\\y  " + "".join(f"{y:7.2f}" for y in ys)
print(hdr, flush=True)
for x in xs:
    row = f"{x:7.2f} "
    for y in ys:
        best, _ = go(np.array([x, y, Z]))
        grid[f"{x}_{y}"] = float(best)
        row += f"{best:7.3f}"
    print(row, flush=True)
json.dump({"z": Z, "xs": xs.tolist(), "ys": ys.tolist(), "grid": grid,
           "objects": {k: v.tolist() for k, v in objs.items()}},
          open("grid_probe.json", "w"), indent=2)
print("GRIDDONE", flush=True)
env.close()
