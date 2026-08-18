"""Capture one LIBERO-Safety scene's RGB frame + camera params + GT geometry.

Standalone companion to `libero_safety_capture.py` (which produced
results_tables/libero_safety_scenes.jsonl with init state 0 / 5 settle steps
/ no camera): this script re-creates the SAME env the eval client builds
(`run_guided_libero_safety_eval.py`: level-aware bddl via
`bm.get_task_bddl_file_path_by_level_id`, init states via
`bm.get_task_init_states_by_level_id`, `env.set_init_state(init[ep % len])`,
zero-action settle) and additionally saves what the paper's RGB overlay
figures need:

  <out_dir>/agentview_rgb.png     raw render, 512x512 (or --res), UNrotated
                                  (visual_conditioning.py's client convention
                                  applies `[::-1, ::-1]` before overlay;
                                  saved raw here, rotate at use site)
  <out_dir>/camera_params.json    {"<camera>": {"intrinsic": 3x3,
                                   "extrinsic": 4x4 cam->world}} -- the exact
                                  format visual_conditioning.py's __main__
                                  block loads
  <out_dir>/scene.json            suite/level/task id+name/language/bddl/ep +
                                  obj_pos {name: [x,y,z]} (all GT entities,
                                  same key rule as the client's
                                  `object_positions`) + eef_pos

Env prerequisites (same as slurm/ls_vanilla_tier.slurm):
  conda activate openvla_libero_merged
  export MUJOCO_GL=egl            # osmesa on CPU-only nodes, if available
  export LIBERO_CONFIG_PATH=$ROOT/LIBERO-Safety/.libero_config
  export PYTHONPATH=$ROOT/LIBERO-Safety:$WT/vlm_pipeline

Usage:
  python capture_scene_rgb.py --suite human_safety --level 0 --task 0 \
      --ep 0 --out_dir paper/figs/scene_cache/hs_L0_t0_ep0

`--task` is the suite-global task index (same indexing as
`run_guided_libero_safety_eval.py --task_filter`); `--level` is checked
against the task's own level and the script aborts on mismatch.

Does NOT modify any pipeline file; read-only w.r.t. the benchmark assets.
"""

import argparse
import json
import pathlib

import numpy as np

NUM_STEPS_WAIT = 10  # settle steps (client uses 20; jsonl capture used 5)
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def object_positions(obs):
    """Same key rule as run_guided_libero_safety_eval.object_positions."""
    return {k[:-4]: [float(x) for x in obs[k]] for k in obs
            if k.endswith("_pos") and not k.startswith("robot0")
            and "_to_" not in k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--task", type=int, required=True,
                    help="suite-global task index (bm.get_task(i))")
    ap.add_argument("--ep", type=int, default=0,
                    help="init-state index: env.set_init_state(init[ep %% len])")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--settle_steps", type=int, default=NUM_STEPS_WAIT)
    args = ap.parse_args()

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )
    import imageio.v2 as imageio

    bm = benchmark.get_benchmark_dict()[args.suite]()
    task = bm.get_task(args.task)
    if getattr(task, "level", None) != args.level:
        raise SystemExit(f"task {args.task} has level {task.level}, "
                         f"not requested L{args.level}")

    bddl = bm.get_task_bddl_file_path_by_level_id(task.level, task.level_id)
    env = OffScreenRenderEnv(bddl_file_name=bddl,
                             camera_heights=args.res, camera_widths=args.res)
    obs = env.reset()
    try:
        init = bm.get_task_init_states_by_level_id(task.level, task.level_id)
        obs = env.set_init_state(init[args.ep % len(init)])
    except FileNotFoundError:
        print("WARNING: no init states shipped for this task; using reset()")
    for _ in range(args.settle_steps):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_key = f"{args.camera}_image"
    if img_key not in obs:
        raise SystemExit(f"no {img_key} in obs; keys: "
                         f"{[k for k in obs if 'image' in k]}")
    imageio.imwrite(out / f"{args.camera}_rgb.png",
                    np.asarray(obs[img_key], dtype=np.uint8))

    K = get_camera_intrinsic_matrix(env.sim, args.camera, args.res, args.res)
    ext = get_camera_extrinsic_matrix(env.sim, args.camera)
    with open(out / "camera_params.json", "w") as f:
        json.dump({args.camera: {"intrinsic": K.tolist(),
                                 "extrinsic": ext.tolist(),
                                 "height": args.res, "width": args.res}},
                  f, indent=1)

    scene = {
        "suite": args.suite, "level": task.level, "task_id": args.task,
        "task": task.name, "language": str(task.language), "bddl": str(bddl),
        "ep": args.ep, "settle_steps": args.settle_steps,
        "camera": args.camera, "res": args.res,
        "obj_pos": object_positions(obs),
        "eef_pos": [float(x) for x in obs["robot0_eef_pos"]],
    }
    with open(out / "scene.json", "w") as f:
        json.dump(scene, f, indent=1)
    env.close()
    print(f"CAPTURE_OK {args.suite} L{task.level} task{args.task} "
          f"ep{args.ep} -> {out}")
    for k, v in scene["obj_pos"].items():
        print(f"  {k}: {np.round(v, 3).tolist()}")


if __name__ == "__main__":
    main()
