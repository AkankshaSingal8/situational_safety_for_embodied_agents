"""Audit the v2 residual dataset before training on it.

    python audit_v2.py --suite safelibero_spatial --data data_v2/spatial/r0 --scenes data_v2/spatial/scenes_train.npz \
        --ckpt runs_scale/spatial/base/ema_last.pt --pi_port 8000 --n_acted 30 --n_quiet 10 --out runs_v2/audit/spatial

1. label check (every episode): drive_tgt equals executed - base on the executed steps; zero without a sentence.
2. replay (sample): re-run the stored executed actions open loop from the scene's init state. Does the episode
   reproduce task success, no contact with the named obstacle, and the same hits on other obstacles?
3. counterfactual (same sample): run pi0.5 alone (no shield, no residual) closed loop on the same scene, twice.
   How often does pi0.5 alone hit the named obstacle or fail the task where the stored labels succeed safely?
"acted" = episodes where the shield changed at least one chunk; "quiet" = it never did (labels all zero).
"""
import argparse
import glob
import json
import os
import time

import numpy as np


def label_check(files):
    bad, n_drv, n_bo, worst = [], 0, 0, 0.0
    nz_chunks, chunks = 0, 0
    for f in files:
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        E, B, T, L = z["exec"], z["base"], z["drive_tgt"], z["exec_len"]
        chunks += len(L)
        if m["drive"]:
            n_drv += 1
            err = max(float(np.abs(T[c, :L[c]] - (E[c, :L[c], :3] - B[c, :L[c], :3])).max()) for c in range(len(L)))
            worst = max(worst, err)
            nz_chunks += int((np.abs(T).max(axis=(1, 2)) > 0).sum())
            if err > 1e-5:
                bad.append((os.path.basename(f), err))
        else:
            n_bo += 1
            dev_ = max(float(np.abs(E[c, :L[c], :3] - B[c, :L[c], :3]).max()) for c in range(len(L)))
            if np.abs(T).max() > 0 or dev_ > 1e-5:
                bad.append((os.path.basename(f), "base-only label or exec != base"))
    return dict(files=len(files), driven=n_drv, base_only=n_bo, chunks=chunks, nonzero_chunks_driven=nz_chunks,
                worst_err=worst, bad=bad[:20], n_bad=len(bad))


def worker(args):
    wid, items, cfg = args
    import torch
    torch.set_num_threads(1)
    import sl_env
    import collect_v2 as CV
    from controllers import make_runner
    from eval_sl import load_scenes
    sl_env.set_suite(cfg["suite"])
    sl_env.ENV_SEED, sl_env.SETTLE = cfg["env_seed"], cfg["settle"]
    sl_env.MAX_STEPS = sl_env.MAX_STEPS_BY_SUITE.get(cfg["suite"], 300)
    states, _ = load_scenes(cfg["scenes"])
    r = make_runner(cfg)
    out, envs = [], {}
    for f, group in items:
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        key = (m["level"], m["task"])
        if key not in envs:
            for v in envs.values():
                v[0].close()
            envs = {key: sl_env.make_env(*key, res=cfg["res"])}
        env, lang, _ = envs[key]
        init = states[m["scene"]]
        # --- replay the stored executed actions ---
        env.reset(); obs = env.set_init_state(init); obs = sl_env.settle(env)
        present = sl_env.active_obstacles(obs)
        init_pos = {n: obs[n + "_pos"].copy() for n in present}
        hits, t, success = {}, 0, False
        for c in range(len(z["exec_len"])):
            for j in range(int(z["exec_len"][c])):
                obs, _, done, _ = env.step(z["exec"][c, j].tolist())
                for n in sl_env.collided(obs, init_pos, [n for n in present if n not in hits]):
                    hits[n] = t
                t += 1
                if done:
                    success = True
                    break
            if success:
                break
        named = [sl_env.scene_name(obs, n) for n in m["drive"]]
        rep = dict(success=success, named_hit=any(n in hits for n in named), other_hit=any(n not in named for n in hits),
                   same_hits=sorted(hits) == sorted(m["hits"]), steps=t, log_steps=m["steps"])
        # --- pi0.5 alone on the same scene ---
        cfs = []
        for k in range(cfg["n_cf"]):
            torch.manual_seed(1000 + k)
            o, _ = CV.run_one(env, init, lang, r, cfg, [], [], {})
            cfs.append(dict(success=o["success"], named_hit=any(n in o["hits"] for n in named), any_hit=bool(o["hits"])))
        out.append(dict(file=os.path.basename(f), group=group, level=m["level"], task=m["task"], drive=m["drive"],
                        others=m["others"], log_hits=m["hits"], shielded=(m.get("filter") or {}).get("n_shielded"),
                        replay=rep, pi05=cfs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="pi05")
    ap.add_argument("--pi_port", type=int, default=8000)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--env_seed", type=int, default=0)
    ap.add_argument("--settle", type=int, default=20)
    ap.add_argument("--exec_steps", type=int, default=5)
    ap.add_argument("--flow_steps", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_acted", type=int, default=30)
    ap.add_argument("--n_quiet", type=int, default=10)
    ap.add_argument("--n_cf", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(f"{a.data}/ep_*.npz"))
    t0 = time.time()
    lc = label_check(files)
    print("label check", json.dumps(lc), f"{(time.time() - t0) / 60:.1f} min", flush=True)
    json.dump(lc, open(f"{a.out}/label_check.json", "w"))
    # sample driven episodes by whether the shield acted (from the logs)
    logs = [json.loads(x) for f in glob.glob(f"{a.data}/log_*.jsonl") for x in open(f)]
    kept = {x["scene"]: x for x in logs if x.get("final") and x["keep"] and x["drive"]}
    acted = [s for s, x in kept.items() if (x.get("filter") or {}).get("n_shielded", 0) > 0]
    quiet = [s for s, x in kept.items() if (x.get("filter") or {}).get("n_shielded", 0) == 0]
    rng = np.random.default_rng(a.seed)
    pick = [(s, "acted") for s in rng.choice(acted, min(a.n_acted, len(acted)), replace=False)]
    pick += [(s, "quiet") for s in rng.choice(quiet, min(a.n_quiet, len(quiet)), replace=False)]
    items = [(f"{a.data}/ep_{int(s):05d}.npz", g) for s, g in pick]
    items.sort(key=lambda it: json.loads(str(np.load(it[0])["meta"]))["task"])
    per = int(np.ceil(len(items) / a.workers))
    blocks = [items[j * per:(j + 1) * per] for j in range(a.workers)]
    cfg = vars(a)
    import multiprocessing as mp
    res = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for o in pool.imap_unordered(worker, [(j, b, cfg) for j, b in enumerate(blocks) if b]):
            res += o
            print(f"{len(res)}/{len(items)} done {(time.time() - t0) / 60:.1f} min", flush=True)
    with open(f"{a.out}/episodes.jsonl", "w") as fo:
        for o in res:
            fo.write(json.dumps(o) + "\n")
    summ = {}
    for g in ("acted", "quiet"):
        rs = [o for o in res if o["group"] == g]
        if not rs:
            continue
        cf = [c for o in rs for c in o["pi05"]]
        summ[g] = dict(n=len(rs),
                       replay_success=np.mean([o["replay"]["success"] for o in rs]),
                       replay_named_hit=np.mean([o["replay"]["named_hit"] for o in rs]),
                       replay_other_hit=np.mean([o["replay"]["other_hit"] for o in rs]),
                       replay_same_hits=np.mean([o["replay"]["same_hits"] for o in rs]),
                       pi05_success=np.mean([c["success"] for c in cf]),
                       pi05_named_hit=np.mean([c["named_hit"] for c in cf]),
                       pi05_any_hit=np.mean([c["any_hit"] for c in cf]),
                       pi05_safe_success=np.mean([c["success"] and not c["named_hit"] for c in cf]))
        summ[g] = {k: (round(float(v), 3) if not isinstance(v, int) else v) for k, v in summ[g].items()}
    print("summary", json.dumps(summ), flush=True)
    json.dump(summ, open(f"{a.out}/summary.json", "w"))


if __name__ == "__main__":
    main()
