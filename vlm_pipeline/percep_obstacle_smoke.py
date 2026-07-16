"""Live-sim smoke test for percep_obstacle.estimate_obstacle_pos (gt_seg leg).

Creates SafeLIBERO spatial envs (L1, first 2 tasks, 3 init states each),
estimates the active obstacle position from RGB-D renders, and reports the
error vs the sim GT — the runtime analogue of the offline localization floor.
CPU-only (osmesa/egl), no policy, no GPU needed.
"""

import json
import sys

import numpy as np

sys.path.insert(0, "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/flow-guidance-tier-gt/vlm_pipeline")
from percep_obstacle import estimate_obstacle_pos  # noqa: E402


def main():
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path

    bm = benchmark.get_benchmark_dict()["safelibero_spatial"]()
    errs = []
    for task_id in range(2):
        task = bm.get_task(task_id)
        init_states = bm.get_task_init_states(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}",
            camera_heights=256, camera_widths=256,
        )
        env.reset()
        for ep in range(3):
            env.set_init_state(init_states[ep])
            for _ in range(10):
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
            names = [n.replace("_joint0", "") for n in env.sim.model.joint_names
                     if "obstacle" in n]
            obstacle = None
            for n in names:
                p = obs.get(f"{n}_pos")
                if p is not None and p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
                    obstacle = n
                    break
            if obstacle is None:
                print(f"task {task_id} ep {ep}: no active obstacle")
                continue
            gt = np.asarray(obs[f"{obstacle}_pos"])
            for cams in (("agentview",), ("agentview", "birdview")):
                est, nv = estimate_obstacle_pos(env.sim, obstacle, cameras=cams)
                err = float(np.linalg.norm(est - gt)) if est is not None else None
                print(json.dumps({"task": task_id, "ep": ep, "obstacle": obstacle,
                                  "cams": list(cams), "views_used": nv,
                                  "err_m": round(err, 4) if err is not None else None,
                                  "gt": gt.round(3).tolist(),
                                  "est": est.round(3).tolist() if est is not None else None}))
                if err is not None and len(cams) == 2:
                    errs.append(err)
        env.close()
    if errs:
        print(f"SMOKE SUMMARY: n={len(errs)} median={np.median(errs):.4f} "
              f"max={np.max(errs):.4f}")
    print("PERCEP_SMOKE_COMPLETE")


if __name__ == "__main__":
    main()
