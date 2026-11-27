"""Residual data and test scenes with Google Scanned Objects obstacles (gso_objects.py).

    # training data: pi0.5 LIBERO + lookahead shield, one GSO obstacle per scene, kept only on safe success
    python collect_gso.py --suite safelibero_spatial --split train --layouts 0-39 --per_layout 4 \
        --out data_gso/spatial --pi_port 8001
    # test scenes (no rollouts): held-out objects on test layouts 40-49, for eval_sl --scenes
    python collect_gso.py --suite safelibero_spatial --split test --layouts 40-49 --per_layout 3 \
        --scenes_only --out data_gso/spatial/scenes_test.npz

Item i = (level, task, layout, k). Its object is the (i mod n)-th of a fixed shuffle of the split's objects, so
each job covers the object list evenly and a resubmission repeats exactly the same items. Per item: build the
env with the object in the first obstacle slot, place it upright with a random yaw by scenes.make_scene (near
the object, on its path or between gripper and object; on the level-II platform with the usual probability),
retry the placement up to 8 times, then run collect_v2.run_one with the obstacle as the driving (named) one.
Episode npz layout as collect_v2 plus meta keys gso, noun, category.
"""
import argparse
import json
import os
import time

import numpy as np


def read_log_lines(path):
    """Parsed lines of one jsonl log; skips lines garbled by concurrent appends (older runs shared log names)."""
    out = []
    for x in open(path, errors="replace"):
        try:
            out.append(json.loads(x.replace("\x00", "").strip()))
        except ValueError:
            pass
    return out


def _with_prompt(f, lang, sentence):
    """Reuse one ex2 websocket client with a new prompt."""
    from residual_pg_ctrl import prompt
    import cv2
    from collect_v2 import state8
    p = prompt(lang, sentence)

    def g(obs, A):
        ag = cv2.resize(obs["agentview_image"], (224, 224), interpolation=cv2.INTER_AREA)
        wr = cv2.resize(obs["robot0_eye_in_hand_image"], (224, 224), interpolation=cv2.INTER_AREA)
        return np.asarray(f.client.ws_call(dict(agent=ag, wrist=wr, prompt=p, state=state8(obs),
                                                base=np.asarray(A, np.float32)))["delta"])
    return g


def items(levels, tasks, layouts, per_layout):
    return [(lv, t, e, k) for lv in levels for t in tasks for e in layouts for k in range(per_layout)]


def worker(args):
    wid, todo, cfg = args
    import torch
    torch.set_num_threads(1)
    import gso_objects as G
    import scenes as SC
    import sl_env
    sl_env.set_suite(cfg["suite"])
    sl_env.ENV_SEED, sl_env.SETTLE = cfg["env_seed"], cfg["settle"]
    G.enable()
    idx = G.load_index()
    r = None
    if not cfg["scenes_only"]:
        from collect_v2 import run_one, save
        from controllers import make_runner
        r = make_runner(cfg)
        grid = json.load(open(cfg["grid"]))
        cells = json.load(open(cfg["cell_cfgs"]))
    os.makedirs(cfg["log_dir"], exist_ok=True)
    log = open(os.path.join(cfg["log_dir"], f"log_{wid}_{os.environ.get('SLURM_JOB_ID', 'local')}.jsonl"), "a")
    done = set()
    for f in os.listdir(cfg["log_dir"]):
        if f.startswith("log_"):
            done |= {x["item"] for x in read_log_lines(os.path.join(cfg["log_dir"], f)) if x.get("final")}
    states, metas = [], []
    exc = None
    for i, (lv, t, e, k), obj in todo:
        if i in done:
            continue
        rng = np.random.default_rng(cfg["seed"] * 1_000_003 + i)
        line = dict(item=i, level=lv, task=t, layout=e, k=k, gso=obj, noun=idx[obj]["noun"],
                    category=idx[obj]["category"])
        env = None
        try:
            env, lang, inits = G.make_env(lv, t, obj, res=cfg["res"])
            sc = None
            lays = [e]
            if cfg.get("balanced"):   # keep every object's run count equal: fall back to other layouts of the range
                lo, hi = map(int, cfg["layouts"].split("-"))
                lays = [lo + (e - lo + 7 * j) % (hi - lo + 1) for j in range(hi - lo + 1)]   # every layout, once
            for e2 in lays:
                for _ in range(8):
                    sc = SC.make_scene(env, inits[e2], rng, 1, types=[G.NAME],
                                       p_elevated=cfg["p_elev_II"] if lv == "II" else cfg["p_elev_I"],
                                       quats={G.NAME: G.yaw_quat(rng)})
                    if sc is not None:
                        break
                if sc is not None:
                    e = e2
                    line["layout"] = e
                    break
            if sc is None:
                line.update(final=True, keep=False, reject="placement")
            elif cfg["scenes_only"]:
                states.append(sc[0])
                metas.append(dict(suite=cfg["suite"], level=lv, task=t, layout=e, obstacles=[G.NAME],
                                  elevated=bool(sc[2]), lang=lang, gso=obj, noun=idx[obj]["noun"],
                                  category=idx[obj]["category"]))
                line.update(final=True, keep=True)
            else:
                names = cells.get(f"{lv}/{t}", ["default"])[: cfg["tries"]]
                if cfg.get("retry_cfgs"):
                    names = cfg["retry_cfgs"].split(",")
                exf = None
                if cfg.get("ex_port"):   # DAgger: ex2 drives (shield-guarded), the shield labels
                    import texts as TX
                    from residual_ex_ctrl import make_exf
                    sent = TX.T_TRAIN[rng.integers(len(TX.T_TRAIN))].format(o=idx[obj]["noun"])
                    if exc is None:
                        exc = make_exf(cfg["ex_port"], lang, sent)
                    exf = _with_prompt(exc, lang, sent)
                    line["sentence"] = sent
                    names = names[:1]
                for k_try, name in enumerate(names):
                    out, rec = run_one(env, sc[0], lang, r, cfg, [G.NAME], [], grid.get(name, {}), exf=exf)
                    # DAgger keeps every episode without a collision: labels are per-state, task outcome is recorded
                    keep = bool(out["safe_success"]) if exf is None else bool(rec is not None and not out["named_hit"])
                    if keep or k_try == len(names) - 1:
                        line.update(final=True, keep=keep, cfg=name, attempt=k_try, elevated=bool(sc[2]),
                                    **{x: out.get(x) for x in ("success", "named_hit", "steps", "filter", "ex_frac")},
                                 dagger=bool(exf))
                        if keep:
                            save(os.path.join(cfg["out"], f"ep_{i:06d}.npz"), rec,
                                 dict(line, lang=lang, drive=[G.NAME], others=[], exec_steps=cfg["exec_steps"],
                                      suite=cfg["suite"], scene=i))
                        break
        except Exception as ex:  # noqa: BLE001
            import traceback
            line.update(final=True, keep=False, error=repr(ex)[:300], tb=traceback.format_exc()[-600:])
        finally:
            if env is not None:
                env.close()
        log.write(json.dumps(line) + "\n"); log.flush()
    return states, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--levels", default="I,II")
    ap.add_argument("--tasks", default="0,1,2,3")
    ap.add_argument("--layouts", default="0-39")
    ap.add_argument("--per_layout", type=int, default=4)
    ap.add_argument("--item_ids", default="", help="lo-hi subset of items (for splitting a suite over jobs)")
    ap.add_argument("--scenes_only", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--base", default="pi05")
    ap.add_argument("--pi_port", type=int, default=8000)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--env_seed", type=int, default=0)
    ap.add_argument("--settle", type=int, default=20)
    ap.add_argument("--exec_steps", type=int, default=5)
    ap.add_argument("--flow_steps", type=int, default=10)
    ap.add_argument("--grid", default="filter_grid.json")
    ap.add_argument("--cell_cfgs", default="")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--p_elev_I", type=float, default=0.2)
    ap.add_argument("--p_elev_II", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ex_port", type=int, default=0, help="DAgger: res_server.py port; ex2 drives, the shield labels")
    ap.add_argument("--balanced", type=int, default=0, help="scenes_only: this many scenes per object and level")
    ap.add_argument("--objects_file", default="", help="json list of GSO model names to use instead of the whole split")
    ap.add_argument("--retry_from", default="", help="earlier output dir (same --seed, --per_layout, --split): rerun "
                    "only its items that were placed but ended without a safe success, writing to --out")
    ap.add_argument("--retry_cfgs", default="shield_h7,shield_h5,shield_wide_h7,shield_h7")
    a = ap.parse_args()
    import gso_objects as G
    G.ensure_local()
    objs = G.objects(a.split)
    if a.objects_file:
        keep = set(json.load(open(a.objects_file)))
        objs = [o for o in objs if o in keep]
    np.random.default_rng(12345).shuffle(objs)
    lo, hi = map(int, a.layouts.split("-"))
    its = items(a.levels.split(","), [int(x) for x in a.tasks.split(",")], range(lo, hi + 1), a.per_layout)
    # objects interleave across suites: offset by suite so the four suites do not start on the same object
    off = ["safelibero_spatial", "safelibero_object", "safelibero_goal", "safelibero_long"].index(a.suite) * 997
    todo = [(i, it, objs[(i + off) % len(objs)]) for i, it in enumerate(its)]
    if a.balanced:
        # every object gets exactly --balanced scenes per level; tasks cycle, layouts drawn from --layouts
        brng = np.random.default_rng(a.seed + 777)
        tasks = [int(x) for x in a.tasks.split(",")]
        its = [(lv, tasks[(j * a.balanced + r) % len(tasks)], int(brng.integers(lo, hi + 1)), r, o)
               for lv in a.levels.split(",") for j, o in enumerate(sorted(objs)) for r in range(a.balanced)]
        todo = [(i, it[:4], it[4]) for i, it in enumerate(its)]
    if a.item_ids:
        i0, i1 = map(int, a.item_ids.split("-"))
        todo = [x for x in todo if i0 <= x[0] <= i1]
    if a.retry_from:
        fin = {}
        for f in os.listdir(a.retry_from):
            if f.startswith("log_"):
                for x in open(os.path.join(a.retry_from, f), errors="replace"):
                    try:
                        x = json.loads(x.replace("\x00", "").strip())
                    except ValueError:
                        continue
                    if x.get("final"):
                        fin[x["item"]] = x
        # the object comes from the earlier log (round 0 used an older, smaller object index); the item tuple and
        # the placement rng depend only on the item number, --seed and --per_layout
        todo = [(i, it, fin[i]["gso"]) for i, it, _ in todo
                if i in fin and not fin[i]["keep"] and fin[i].get("reject") != "placement"]
        bad = [x for x in todo if (fin[x[0]]["level"], fin[x[0]]["task"], fin[x[0]]["layout"], fin[x[0]]["k"]) != x[1]]
        assert not bad, f"retry items do not match {a.retry_from} (wrong --per_layout?): {bad[:3]}"
        print(f"retry: {len(todo)} items without a safe success in {a.retry_from}", flush=True)
    else:
        a.retry_cfgs = ""
    cfg = vars(a)
    if a.scenes_only:
        cfg["log_dir"] = os.path.splitext(a.out)[0] + "_log"
    else:
        os.makedirs(a.out, exist_ok=True)
        cfg["log_dir"] = a.out
    print(f"{a.suite} {a.split}: {len(objs)} objects, {len(todo)} items", flush=True)
    blocks = [todo[j::a.workers] for j in range(a.workers)]
    t0 = time.time()
    import multiprocessing as mp
    S, M = [], []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for st, me in pool.imap_unordered(worker, [(j, b, cfg) for j, b in enumerate(blocks) if b]):
            S += st; M += me
            print(f"worker done {(time.time() - t0) / 60:.1f} min", flush=True)
    if a.scenes_only:
        order = sorted(range(len(M)), key=lambda j: (M[j]["level"], M[j]["task"], M[j]["layout"], M[j]["gso"]))
        arr = np.empty(len(S), dtype=object)
        for j, o in enumerate(order):
            arr[j] = S[o]
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        np.savez_compressed(a.out, states=arr, meta=json.dumps([M[o] for o in order]))
        print("wrote", a.out, len(S))
    rows = [x for f in os.listdir(cfg["log_dir"]) if f.startswith("log_")
            for x in read_log_lines(os.path.join(cfg["log_dir"], f))]
    fin = [x for x in rows if x.get("final")]
    print(f"items {len(fin)}: kept {sum(x['keep'] for x in fin)}, placement rejects "
          f"{sum(x.get('reject') == 'placement' for x in fin)}, errors {sum('error' in x for x in fin)}", flush=True)


if __name__ == "__main__":
    main()
