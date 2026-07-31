"""CPU-only SafeLIBERO scene capture for offline identity benchmarking.

For every suite x safety level x task x first N init states, constructs the
env the same way run_guided_safelibero_pi05_eval.py does (same bddl-file
resolution, seed protocol, init-state application, NUM_STEPS_WAIT dummy
settle steps, and gt_obstacle_name derivation), but WITHOUT any camera
rendering (ControlEnv with use_camera_obs=False, has_offscreen_renderer=False
-- no EGL/GPU needed) and without ever contacting a policy server.

One JSONL record per (suite, level, task, init_idx):
  {suite, level, task_id, task_description, init_idx, gt_obstacle_name,
   eef_pos, objects: {name: pos}, objects_all: {name: pos}}

`objects` mirrors the client's symbolic-identity candidate filter
(identify_obstacle's _cands: *_pos keys, not robot0, no '_to_', shape (3,),
workspace (-0.5,0.5)x(-0.5,0.5), z>0). `objects_all` is the unfiltered set
(the eval client's fol path builds candidates WITHOUT the workspace filter),
so the offline bench can reproduce both entity views exactly.

Usage:
  python vlm_pipeline/ident_scene_capture.py \
      --out results_tables/safelibero_ident_scenes.jsonl
"""

import argparse
import json
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from libero.libero import benchmark  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402

# Reuse the eval client's constants (seed protocol / settle window). Importing
# the module pulls in openpi_client (policy-server dep) which may be absent in
# a capture-only env — fall back to verbatim copies of the two constants.
try:
    from run_guided_safelibero_pi05_eval import DUMMY_ACTION, NUM_STEPS_WAIT  # noqa: E402
except ImportError as _e:  # openpi_client not installed: constants copied verbatim
    logging.warning(f"run_guided import failed ({_e}); using verbatim constant copies")
    NUM_STEPS_WAIT = 20
    DUMMY_ACTION = [0.0] * 6 + [-1.0]

SUITES = ["safelibero_spatial", "safelibero_goal", "safelibero_object", "safelibero_long"]
LEVELS = ["I", "II"]


def _get_cpu_env(task, seed):
    """Same bddl resolution as run_guided_..._get_libero_env, but a pure-CPU
    ControlEnv: no cameras, no offscreen renderer (OffScreenRenderEnv only
    differs by forcing has_offscreen_renderer=True + camera obs)."""
    # ControlEnv is defined in env_wrapper but not re-exported by
    # libero.libero.envs.__init__ in the SafeLIBERO fork.
    from libero.libero.envs.env_wrapper import ControlEnv

    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = ControlEnv(
        bddl_file_name=str(task_bddl_file),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        camera_depths=False,
    )
    env.seed(seed)
    return env, task_description


def _sim_of(env):
    if hasattr(env, "sim"):
        return env.sim
    return env.env.sim


def _gt_obstacle_name(env, obs):
    """Verbatim logic from run_guided_safelibero_pi05_eval.run_eval."""
    sim = _sim_of(env)
    obstacle_names = [
        n.replace("_joint0", "") for n in sim.model.joint_names if "obstacle" in n
    ]
    for name in obstacle_names:
        p = obs.get(f"{name}_pos", None)
        if p is not None and p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
            return name
    return None


def _object_views(obs, workspace=((-0.5, 0.5), (-0.5, 0.5))):
    """(filtered, unfiltered) candidate dicts.

    filtered  = identify_obstacle's _cands filter (symbolic entity view);
    unfiltered = the eval client's fol _cands (no workspace/shape gate beyond
    the key rules; non-(3,) values are skipped and logged).
    """
    filtered, allv = {}, {}
    for k in obs:
        if not k.endswith("_pos") or k.startswith("robot0"):
            continue
        if "_to_" in k:
            continue
        p = np.asarray(obs[k])
        if p.shape != (3,):
            logging.warning(f"skipping non-3-vector obs key {k} shape={p.shape}")
            continue
        allv[k[:-4]] = p.tolist()
        if (workspace[0][0] < p[0] < workspace[0][1]
                and workspace[1][0] < p[1] < workspace[1][1] and p[2] > 0):
            filtered[k[:-4]] = p.tolist()
    return filtered, allv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default=str(pathlib.Path(__file__).parents[1]
                                / "results_tables/safelibero_ident_scenes.jsonl"))
    ap.add_argument("--num_init_states", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)  # eval client default
    ap.add_argument("--suites", type=str, nargs="+", default=SUITES)
    ap.add_argument("--levels", type=str, nargs="+", default=LEVELS)
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, "w")

    benchmark_dict = benchmark.get_benchmark_dict()
    n_records = 0
    counts = {}

    for suite in args.suites:
        for level in args.levels:
            try:
                task_suite = benchmark_dict[suite](safety_level=level)
            except Exception as e:  # noqa: BLE001
                logging.error(f"SUITE BUILD FAILED {suite} level {level}: {e}")
                continue
            for task_id in range(task_suite.n_tasks):
                task = task_suite.get_task(task_id)
                try:
                    initial_states = task_suite.get_task_init_states(task_id)
                    env, task_description = _get_cpu_env(task, args.seed)
                except Exception as e:  # noqa: BLE001
                    logging.error(f"ENV BUILD FAILED {suite}/{level}/t{task_id}: {e}")
                    continue
                logging.info(f"=== {suite} level {level} task {task_id}: {task_description}")
                cell_n = 0
                for init_idx in range(min(args.num_init_states, len(initial_states))):
                    try:
                        env.reset()
                        obs = env.set_init_state(initial_states[init_idx])
                        for _ in range(NUM_STEPS_WAIT):
                            obs, _, _, _ = env.step(DUMMY_ACTION)
                        gt = _gt_obstacle_name(env, obs)
                        objects, objects_all = _object_views(obs)
                        rec = {
                            "suite": suite,
                            "level": level,
                            "task_id": int(task_id),
                            "task_description": str(task_description),
                            "init_idx": int(init_idx),
                            "gt_obstacle_name": gt,
                            "eef_pos": np.asarray(obs["robot0_eef_pos"]).tolist(),
                            "objects": objects,
                            "objects_all": objects_all,
                        }
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        n_records += 1
                        cell_n += 1
                    except Exception as e:  # noqa: BLE001
                        logging.error(f"EPISODE FAILED {suite}/{level}/t{task_id}/i{init_idx}: {e}")
                counts[f"{suite}/{level}/t{task_id}"] = cell_n
                env.close()

    fh.close()
    logging.info(f"DONE: {n_records} records -> {out_path}")
    for k, v in counts.items():
        logging.info(f"  {k}: {v}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
