"""Measure which transit endpoints the arm can actually hold, and at what height.

Answers two questions the trajectories raised:
  1. Can the end effector reach candidate goal positions at transit height?
  2. Does it hold height while translating, or sag toward the table?
"""
import os, json, time
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
plate = np.asarray(obs["plate_1_pos"])[:3]
print("plate", np.round(plate, 3), "eef0", np.round(obs["robot0_eef_pos"], 3), flush=True)

def go(target, steps=250):
    env.seed(1000); o = env.reset()
    best = 1e9; traj = []
    for i in range(steps):
        p = np.asarray(o["robot0_eef_pos"])[:3]
        traj.append(p.copy())
        best = min(best, float(np.linalg.norm(p - target)))
        o, _, _, _ = env.step(waypoint_action(p, target))
    p = np.asarray(o["robot0_eef_pos"])[:3]
    return best, p, np.array(traj)

results = {}
for dz in (0.04, 0.10, 0.16):
    for L in (0.12, 0.18, 0.24):
        for sign, nm in ((+1, "+x"), (-1, "-x")):
            t = plate + np.array([sign * L, 0.0, dz])
            best, final, traj = go(t)
            key = f"dz{dz}_L{L}_{nm}"
            results[key] = {"target": t.tolist(), "best": best,
                            "final": final.tolist(),
                            "min_z": float(traj[:, 2].min())}
            print(f"{key:20s} target={np.round(t,3)} best={best:.3f} "
                  f"final={np.round(final,3)} minz={traj[:,2].min():.3f}", flush=True)
json.dump(results, open("reach_probe.json", "w"), indent=2)
print("REACHDONE", flush=True)
env.close()
