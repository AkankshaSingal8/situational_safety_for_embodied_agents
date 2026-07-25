"""LS Tier-Percep offline validation (gates the no-GT rollout job).

Per (suite, task) on obstacle_avoidance / human_safety /
obstacle_avoidance_human: build env (L0-L2, init state 0, settle), run the
GroundingDINO+depth back-projection stack on all scene objects, and score:
  - localization rate + per-object position error vs GT obs positions
  - ident transfer: fol / scored top-1 on PERCEP candidates vs the
    CheckRobotContact hazard GT (the exact runtime no-GT condition)
GPU required (GroundingDINO). Output: results_tables/ls_percep_validation.json
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

SUITES = ["obstacle_avoidance", "human_safety", "obstacle_avoidance_human"]
OUT = pathlib.Path(__file__).parents[1] / "results_tables/ls_percep_validation.json"
WORKSPACE = ((-0.8, 0.8), (-0.8, 0.8))


def main():
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from percep_obstacle import make_gdino_detector, estimate_object_positions
    from symbolic_identity import fol_obstacle_id, scored_obstacle_id, load_vlm_priors
    from run_guided_libero_safety_eval import hazard_names_from_constraints

    priors = pathlib.Path(__file__).parents[1] / "results_tables/vlm_hazard_priors.json"
    load_vlm_priors(str(priors), floor=0.5)
    det = make_gdino_detector()
    rows, agg = [], {}
    for suite in SUITES:
        bm = benchmark.get_benchmark_dict()[suite]()
        for ti in range(bm.get_num_tasks()):
            task = bm.get_task(ti)
            try:
                env = OffScreenRenderEnv(
                    bddl_file_name=bm.get_task_bddl_file_path_by_level_id(
                        task.level, task.level_id),
                    camera_heights=256, camera_widths=256)
                obs = env.reset()
                try:
                    init = bm.get_task_init_states_by_level_id(task.level, task.level_id)
                    obs = env.set_init_state(init[0])
                except FileNotFoundError:
                    pass
                for _ in range(5):
                    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                gt = {k[:-4]: np.asarray(obs[k]) for k in obs
                      if k.endswith("_pos") and not k.startswith("robot0")
                      and "_to_" not in k}
                est = estimate_object_positions(env.sim, list(gt),
                                                cameras=("agentview",),
                                                region_source="detector",
                                                detector=det)
                errs = {n: float(np.linalg.norm(est[n] - gt[n]))
                        for n in est}
                cands = {k: v for k, v in est.items()
                         if WORKSPACE[0][0] < v[0] < WORKSPACE[0][1]
                         and WORKSPACE[1][0] < v[1] < WORKSPACE[1][1]
                         and v[2] > 0}
                hazards = hazard_names_from_constraints(env)
                eef = np.asarray(obs["robot0_eef_pos"])
                desc = str(task.language)
                fol = fol_obstacle_id(desc, cands, eef)[:1] if cands else []
                sym = scored_obstacle_id(desc, cands, eef) if cands else None
                row = {"suite": suite, "task": task.name, "level": task.level,
                       "n_obj": len(gt), "n_loc": len(est),
                       "med_err": float(np.median(list(errs.values()))) if errs else None,
                       "hazards": sorted(hazards),
                       "fol_hit": bool(fol) and fol[0] in hazards if hazards else None,
                       "sym_hit": (sym in hazards) if hazards else None,
                       "errs": {k: round(v, 3) for k, v in errs.items()}}
                env.close()
            except Exception as e:  # noqa: BLE001 — keep sweeping, log the scene
                row = {"suite": suite, "task": task.name,
                       "fail": f"{type(e).__name__}: {e}"}
            rows.append(row)
            print(json.dumps(row), flush=True)
            a = agg.setdefault(suite, {"n": 0, "loc": 0, "tot": 0,
                                       "fol": 0, "sym": 0, "nid": 0, "errs": []})
            if "fail" not in row:
                a["n"] += 1
                a["loc"] += row["n_loc"]
                a["tot"] += row["n_obj"]
                a["errs"] += list(row["errs"].values())
                if row["fol_hit"] is not None:
                    a["nid"] += 1
                    a["fol"] += int(row["fol_hit"])
                    a["sym"] += int(row["sym_hit"])
    summary = {s: {"scenes": a["n"], "loc_rate": round(a["loc"] / max(a["tot"], 1), 3),
                   "med_err": round(float(np.median(a["errs"])), 3) if a["errs"] else None,
                   "fol_top1": f"{a['fol']}/{a['nid']}",
                   "sym_top1": f"{a['sym']}/{a['nid']}"}
               for s, a in agg.items()}
    json.dump({"summary": summary, "rows": rows}, open(OUT, "w"), indent=1)
    print("SUMMARY " + json.dumps(summary))
    print("LS_PERCEP_VALIDATE_COMPLETE")


if __name__ == "__main__":
    main()
