"""Live-sim identity smoke: symbolic_obstacle_id vs GT across ALL suites.

The v17 grounder was validated 21/21 (and 39/39 offline) on safelibero_spatial
only. This runs every suite x level x task x 3 init states, no policy, and
reports per-suite identity accuracy — the coverage number Phase 5 needs before
running guided no-GT arms on Goal/Object/Long.
"""

import json
import os
import sys

import numpy as np

EP_START = int(os.environ.get("EP_START", "0"))
EP_END = int(os.environ.get("EP_END", "3"))  # exclusive

sys.path.insert(0, "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/flow-guidance-tier-gt/vlm_pipeline")
from symbolic_identity import identify_obstacle  # noqa: E402


def main():
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path

    totals = {}
    for suite in ["safelibero_spatial", "safelibero_goal", "safelibero_object",
                  "safelibero_long"]:
        bm = benchmark.get_benchmark_dict()[suite]()
        ok = n = 0
        for task_id in range(bm.get_num_tasks()):
            task = bm.get_task(task_id)
            init_states = bm.get_task_init_states(task_id)
            try:
                env = OffScreenRenderEnv(
                    bddl_file_name=f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}",
                    camera_heights=128, camera_widths=128,
                )
            except Exception as e:
                print(f"{suite} t{task_id}: env failed: {e}")
                continue
            env.reset()
            desc = task.language
            for ep in range(EP_START, min(EP_END, len(init_states))):
                obs = env.set_init_state(init_states[ep])
                for _ in range(10):
                    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                gt = None
                for jn in env.sim.model.joint_names:
                    if "obstacle" in jn:
                        name = jn.replace("_joint0", "")
                        p = obs.get(f"{name}_pos")
                        if p is not None and p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
                            gt = name
                            break
                if gt is None:
                    continue
                picked = identify_obstacle(str(desc), obs)
                n += 1
                ok += int(picked == gt)
                if picked != gt:
                    print(json.dumps({"suite": suite, "task": task_id, "ep": ep,
                                      "gt": gt, "picked": picked, "desc": str(desc)}))
            env.close()
        totals[suite] = (ok, n)
        print(f"{suite}: identity {ok}/{n}")
    print("IDENTITY_SMOKE_COMPLETE", json.dumps({k: f"{a}/{b}" for k, (a, b) in totals.items()}))


if __name__ == "__main__":
    main()
