"""Does the protected object move on its own, with the robot held still?"""
import os
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

bm = benchmark.get_benchmark_dict()["safelibero_spatial"]()
task = bm.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = ControlEnv(bddl, use_camera_obs=False, has_offscreen_renderer=False,
                 has_renderer=False, camera_names=[], control_freq=20,
                 hard_reset=False, horizon=100000, ignore_done=True)
KEYS = ["glazed_rim_porcelain_ramekin_1_pos", "plate_1_pos",
        "akita_black_bowl_1_pos", "akita_black_bowl_2_pos", "cookies_1_pos"]
for seed in (2000, 2003, 2007):
    env.seed(seed); obs = env.reset()
    p0 = {k: np.asarray(obs[k])[:3].copy() for k in KEYS}
    print(f"\n--- seed {seed} (robot commanded to hold still) ---", flush=True)
    for step in range(1, 81):
        obs, _, _, _ = env.step(np.zeros(7))
        if step in (1, 5, 10, 20, 40, 80):
            line = f"  t={step:3d} "
            for k in KEYS:
                d = float(np.linalg.norm(np.asarray(obs[k])[:3] - p0[k]))
                line += f"{k.split('_1_')[0][:12]}={d:.4f} "
            print(line, flush=True)
print("\nOBJDONE", flush=True)
env.close()
