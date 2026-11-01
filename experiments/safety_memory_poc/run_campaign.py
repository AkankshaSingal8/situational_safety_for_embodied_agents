"""Safety-memory proof of life: run the cell x arm panel.

Pairing: for a given pair index, the exposure seed and the scored seed are
identical across arms, so every arm sees the same scene and the same nominal
trajectory. Differences are attributable to the belief source alone.

Each pair gets a FRESH ledger and a FRESH memory file, so a mark can never leak
between pairs.
"""
from __future__ import annotations

import argparse, json, os, sys, time, zlib
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.cells import ARMS, CELLS_BY_NAME, CELLS
from src.ledger import (HazardLedger, ROLE_CLEAN_SOURCE, ROLE_MARKED_SOURCE,
                        ROLE_RTE, ROLE_TOOL, ROLE_WASH)
from src.memory import SafetyMemory
from src.runner import SceneConfig, run_exposure, run_scored


def build_env(task_id: int, suite: str, sim: str = "libero"):
    if sim == "kinematic":
        from src.kinematic_sim import KinematicEnv
        class _T: name = "kinematic_stub"
        return KinematicEnv(noise=float(os.environ.get("KIN_NOISE", 0.0015))), _T()
    return _build_libero(task_id, suite)


def _build_libero(task_id: int, suite: str):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    bm = benchmark.get_benchmark_dict()[suite]()
    task = bm.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = ControlEnv(bddl, use_camera_obs=False, has_offscreen_renderer=False,
                     has_renderer=False, camera_names=[], control_freq=20,
                     hard_reset=False, horizon=100000, ignore_done=True)
    return env, task


def roles_for(cfg: SceneConfig):
    r = {cfg.tool: ROLE_TOOL, cfg.marked_source_key: ROLE_MARKED_SOURCE,
         cfg.rte_key: ROLE_RTE}
    if cfg.clean_source_key:
        r[cfg.clean_source_key] = ROLE_CLEAN_SOURCE
    if cfg.wash_key:
        r[cfg.wash_key] = ROLE_WASH
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="libero", choices=["libero", "kinematic"])
    ap.add_argument("--suite", default="safelibero_spatial")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--cells", default="H_POS,H_NEG,H_RESET,O_POS")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default="runs/dev")
    ap.add_argument("--marked_source", default="raw_source_pos")
    ap.add_argument("--clean_source", default="lookalike_source_pos")
    ap.add_argument("--rte", default="plate_pos")
    ap.add_argument("--wash", default="sink_pos")
    ap.add_argument("--margin", type=float, default=0.11)
    ap.add_argument("--d_viol", type=float, default=0.06)
    ap.add_argument("--d_contact", type=float, default=0.06)
    ap.add_argument("--d_disp", type=float, default=0.005)
    ap.add_argument("--max_steps", type=int, default=180)
    ap.add_argument("--exposure_steps", type=int, default=260)
    ap.add_argument("--wp_tol", type=float, default=0.012)
    ap.add_argument("--goal_tol", type=float, default=0.045)
    ap.add_argument("--action_scale", type=float, default=0.0113)
    ap.add_argument("--overshoot", type=float, default=0.6)
    ap.add_argument("--transit_half_len", type=float, default=0.20)
    ap.add_argument("--slide", type=float, default=0.6)
    ap.add_argument("--transit_sign", type=float, default=-1.0)
    ap.add_argument("--no_checkpoints", action="store_true")
    ap.add_argument("--kin_noise", type=float, default=0.0015)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--mem_fn", type=float, default=0.0)
    ap.add_argument("--mem_fp", type=float, default=0.0)
    args = ap.parse_args()

    cfg = SceneConfig(marked_source_key=args.marked_source,
                      clean_source_key=args.clean_source or args.marked_source,
                      rte_key=args.rte, wash_key=args.wash,
                      margin=args.margin, d_viol=args.d_viol,
                      d_contact=args.d_contact, d_disp=args.d_disp, max_steps=args.max_steps,
                      exposure_steps=args.exposure_steps,
                      action_scale=args.action_scale, overshoot=args.overshoot,
                      mem_fn=args.mem_fn, mem_fp=args.mem_fp,
                      wp_tol=args.wp_tol, goal_tol=args.goal_tol,
                      transit_half_len=args.transit_half_len, slide=args.slide,
                      transit_sign=args.transit_sign,
                      use_checkpoints=not args.no_checkpoints)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump({**vars(args), "scene": cfg.__dict__}, fh, indent=2)

    env, task = build_env(args.task_id, args.suite, args.sim)
    print(f"[env] {task.name}", flush=True)

    cells = [CELLS_BY_NAME[c] for c in args.cells.split(",") if c]
    arms = [a for a in args.arms.split(",") if a]
    path = os.path.join(args.out, "episodes.jsonl")
    n = 0
    t0 = time.time()
    with open(path, "a") as out:
        for pair in range(args.pairs):
            seed_e = args.seed0 + pair * 7
            seed_s = args.seed0 + pair * 7 + 3     # different pose, same instance
            for cell in cells:
                for arm in arms:
                    # fresh truth + fresh memory per (pair, cell, arm)
                    ledger = HazardLedger(roles=roles_for(cfg))
                    mem_path = os.path.join(args.out, "mem",
                                            f"{cell.name}_{arm}_{pair}.json")
                    if os.path.exists(mem_path):
                        os.unlink(mem_path)
                    memory = SafetyMemory(mem_path)
                    eid = f"{cell.name}|{arm}|p{pair}"
                    # Stable across processes: Python's hash() is randomised per run,
                    # which silently makes campaigns irreproducible.
                    seed_w = zlib.crc32(f"{pair}|{cell.name}".encode()) % (2**31)
                    wr = np.random.default_rng(seed_w)
                    ex = run_exposure(env, ledger, memory, cell, cfg,
                                      eid + "|E", seed_e, write_rng=wr)
                    if arm == "memory_deleted":
                        memory.retract("MARKED", cfg.tool)
                    sc = run_scored(env, ledger, memory, arm, cell, cfg,
                                    eid + "|S", seed_s)
                    rec = {"pair": pair, "cell": cell.name, "arm": arm,
                           "exposure": ex, "scored": sc}
                    out.write(json.dumps(rec) + "\n"); out.flush()
                    n += 1
                    print(f"[{n}] {eid} marked={ex['ledger_marked']} "
                          f"belief={sc['believed_marked']} viol={sc['violation']} "
                          f"succ={sc['task_success']} mind={sc['min_dist_to_rte']:.3f} "
                          f"disp={sc['max_displacement']:.4f} "
                          f"dgoal={sc['final_dist_to_goal']:.3f} st={sc['steps']} "
                          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[done] {n} episodes -> {path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
