"""Offline probe: does multi-camera fusion improve entity localization?

The no-GT board localizes ALL scene entities with
`estimate_object_positions(sim, names, cameras=("agentview",), ...)` — a
single view. Board floor: med 0.067 m / p90 0.152 m; the F0 attribution
study charged corridor-entity localization error ~12 TSR on
safelibero_object L1. `estimate_obstacle_pos` already fuses across cameras
by mean, so the only question is whether extra static views buy accuracy.

For each safelibero_object scene (levels II+I x 4 tasks x N init states,
settled), run the SAME estimator under camera sets:
  A = ("agentview",)                current board configuration
  B = ("agentview", "birdview")     2-view fusion (the offline-floor setup)
  C = all static workspace cameras  (eye-in-hand excluded: it moves)
and compare against GT obs[f"{name}_pos"]. Entity enumeration mirrors the
eval client (run_guided_safelibero_pi05_eval.py ~L495): obs keys ending in
`_pos`, excluding robot0* and any name containing "_to_". Pure perception —
no policy server, no rollouts.

Output: per-camset {n_attempted, n_localized, med/p90 3D and xy error} plus
a per-object-name breakdown (target vs dest vs obstacle entities are
distinguishable by name).
"""

import argparse
import json
import logging
import pathlib

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from percep_obstacle import estimate_object_positions

logging.basicConfig(level=logging.INFO, format="%(message)s")
SETTLE = 10
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
SEED = 7
Z_CORRECTION = -0.03
MASK_SOURCE = "gt_seg"  # board's --mask_source default in the eval client


def is_static_workspace_cam(name):
    """Keep fixed workspace cameras; drop robot-mounted (moving) ones."""
    return "eye_in_hand" not in name and not name.startswith("robot")


def build_camsets(camera_names):
    """A=current single view, B=+birdview, C=all static views.

    Degrades gracefully: if birdview is absent, B takes the first other
    static views available. Duplicate sets are dropped (keyed by tuple).
    """
    static = [c for c in camera_names if is_static_workspace_cam(c)]
    if "agentview" not in static:
        raise RuntimeError(f"agentview missing from cameras {camera_names}")
    others = [c for c in static if c != "agentview"]
    b_extra = ["birdview"] if "birdview" in others else others[:1]
    camsets = {
        "A_agentview": ("agentview",),
        "B_2view": tuple(["agentview"] + b_extra),
        "C_all_static": tuple(["agentview"] + others),
    }
    seen, out = set(), {}
    for label, cams in camsets.items():
        if cams not in seen and len(cams) >= 1:
            seen.add(cams)
            out[label] = cams
    return out


def scene_entity_names(obs):
    """Mirror the eval client's enumeration (run_guided...eval.py ~L495)."""
    return [k[:-4] for k in obs if k.endswith("_pos")
            and not k.startswith("robot0") and "_to_" not in k]


def summarize(rows):
    att = [r for r in rows]
    loc = [r for r in rows if r["localized"]]
    errs = np.array([r["err_3d"] for r in loc]) if loc else np.array([])
    xy = np.array([r["err_xy"] for r in loc]) if loc else np.array([])

    def q(a, p):
        return float(np.percentile(a, p)) if len(a) else None

    return {
        "n_attempted": len(att),
        "n_localized": len(loc),
        "med_err": q(errs, 50), "p90_err": q(errs, 90),
        "med_xy": q(xy, 50), "p90_xy": q(xy, 90),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", default="II,I",
                    help="comma-separated safety levels (default II,I)")
    ap.add_argument("--episodes_per_task", type=int, default=10)
    ap.add_argument("--out", default=str(
        pathlib.Path(__file__).resolve().parent.parent
        / "entity_mv_probe" / "probe.json"))
    args = ap.parse_args()
    levels = [s.strip() for s in args.levels.split(",") if s.strip()]

    camsets = None
    discovered_cams = []
    rows = []
    for level in levels:
        suite = benchmark.get_benchmark_dict()["safelibero_object"](
            safety_level=level)
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            bddl = (pathlib.Path(get_libero_path("bddl_files"))
                    / task.problem_folder / task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                     camera_heights=256, camera_widths=256,
                                     camera_depths=False)
            env.seed(SEED)
            inits = suite.get_task_init_states(task_id)
            if camsets is None:
                discovered_cams = list(env.sim.model.camera_names)
                logging.info("model camera_names: %s", discovered_cams)
                camsets = build_camsets(discovered_cams)
                logging.info("camera sets: %s",
                             {k: list(v) for k, v in camsets.items()})
            for ep in range(args.episodes_per_task):
                env.reset()
                obs = env.set_init_state(inits[ep % len(inits)])
                for _ in range(SETTLE):
                    obs, *_ = env.step(DUMMY_ACTION)
                names = scene_entity_names(obs)
                if not names:
                    logging.warning("L%s t%d ep%d: no entities, skipping",
                                    level, task_id, ep)
                    continue
                gt = {n: np.asarray(obs[f"{n}_pos"], dtype=np.float64)
                      for n in names}
                for label, cams in camsets.items():
                    est = estimate_object_positions(
                        env.sim, names, cameras=cams,
                        region_source=MASK_SOURCE,
                        z_correction=Z_CORRECTION)
                    for n in names:
                        row = {"level": level, "task_id": task_id, "ep": ep,
                               "object": n, "camset": label,
                               "localized": n in est}
                        if n in est:
                            d = np.asarray(est[n]) - gt[n]
                            row["err_3d"] = float(np.linalg.norm(d))
                            row["err_xy"] = float(np.linalg.norm(d[:2]))
                        rows.append(row)
                done = {label: summarize(
                    [r for r in rows if r["camset"] == label
                     and r["level"] == level and r["task_id"] == task_id
                     and r["ep"] == ep]) for label in camsets}
                logging.info("L%s t%d ep%d (%d objs): %s", level, task_id, ep,
                             len(names),
                             {k: (v["n_localized"], v["med_err"])
                              for k, v in done.items()})
            env.close()

    if camsets is None or not rows:
        raise RuntimeError("probe collected no rows — harness failure, "
                           "refusing to write a summary")

    summary = {}
    for label in camsets:
        cs_rows = [r for r in rows if r["camset"] == label]
        s = summarize(cs_rows)
        per_obj = {}
        for name in sorted({r["object"] for r in cs_rows}):
            per_obj[name] = summarize(
                [r for r in cs_rows if r["object"] == name])
        s["per_object_name"] = per_obj
        summary[label] = s

    hdr = (f"{'camset':<14} {'cams':>4} {'n_att':>6} {'n_loc':>6} "
           f"{'med3d':>7} {'p90_3d':>7} {'med_xy':>7} {'p90_xy':>7}")
    logging.info("\n%s\n%s", hdr, "-" * len(hdr))
    for label, s in summary.items():
        fmt = lambda v: f"{v:7.3f}" if v is not None else "   n/a "
        logging.info("%-14s %4d %6d %6d %s %s %s %s", label,
                     len(camsets[label]), s["n_attempted"], s["n_localized"],
                     fmt(s["med_err"]), fmt(s["p90_err"]),
                     fmt(s["med_xy"]), fmt(s["p90_xy"]))
    logging.info("\nper-object med_err (m) by camset:")
    all_names = sorted({r["object"] for r in rows})
    for name in all_names:
        vals = []
        for label in camsets:
            v = summary[label]["per_object_name"].get(name, {}).get("med_err")
            vals.append(f"{label.split('_')[0]}={v:.3f}" if v is not None
                        else f"{label.split('_')[0]}=miss")
        logging.info("  %-42s %s", name, "  ".join(vals))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"suite": "safelibero_object", "levels": levels,
                   "episodes_per_task": args.episodes_per_task,
                   "seed": SEED, "settle": SETTLE,
                   "mask_source": MASK_SOURCE, "z_correction": Z_CORRECTION},
        "camera_names": discovered_cams,
        "camsets": {k: list(v) for k, v in camsets.items()},
        "summary": summary,
        "rows": rows,
    }, indent=1))
    logging.info("wrote %s (%d rows)", out, len(rows))


if __name__ == "__main__":
    main()
