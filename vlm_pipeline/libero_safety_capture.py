"""Capture LIBERO-Safety scene snapshots for offline scene-graph V2 validation.

Per (suite, task): build env from the level-aware bddl, apply init state 0,
settle 5 steps, dump task language + all object positions + eef pos + bddl
path to one JSONL. Hazard GT comes OFFLINE from the bddl :constraints block
(bddl_utils.robosuite_parse_problem — no env needed), so this job only has to
capture geometry.
"""

import json
import pathlib

SUITES = ["obstacle_avoidance", "human_safety", "affordance", "reasoning_safety"]
OUT = pathlib.Path(__file__).parents[1] / "results_tables/libero_safety_scenes.jsonl"


def main():
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bdict = benchmark.get_benchmark_dict()
    rows = []
    for suite in SUITES:
        bm = bdict[suite]()
        for ti in range(bm.get_num_tasks()):
            task = bm.get_task(ti)
            try:
                bddl = bm.get_task_bddl_file_path_by_level_id(task.level, task.level_id)
                env = OffScreenRenderEnv(bddl_file_name=bddl,
                                         camera_heights=128, camera_widths=128)
                obs = env.reset()
                try:
                    init = bm.get_task_init_states_by_level_id(task.level, task.level_id)
                    obs = env.set_init_state(init[0])
                except FileNotFoundError:
                    pass
                for _ in range(5):
                    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                pos = {k[:-4]: [float(x) for x in obs[k]] for k in obs
                       if k.endswith("_pos") and not k.startswith("robot0") and "_to_" not in k}
                rows.append({
                    "suite": suite, "task": task.name, "level": task.level,
                    "language": str(task.language), "bddl": str(bddl),
                    "obj_pos": pos,
                    "eef_pos": [float(x) for x in obs["robot0_eef_pos"]],
                })
                env.close()
                print(f"ok {suite}/{task.name}")
            except Exception as e:  # noqa: BLE001
                rows.append({"suite": suite, "task": task.name, "fail": f"{type(e).__name__}: {e}"})
                print(f"FAIL {suite}/{task.name}: {e}")
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = sum(1 for r in rows if "fail" not in r)
    print(f"CAPTURE_COMPLETE {ok}/{len(rows)} -> {OUT}")


if __name__ == "__main__":
    main()
