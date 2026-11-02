"""Collect residual training data: base policy + oracle detour, closed loop, on generated scenes.

    python collect_residual.py --ckpt runs/base/ema_last.pt --scenes data/scenes_train.npz \
                               --out data/residual_train --workers 5

Per episode one "driving" sentence decides what is executed: it names a present obstacle
(p=0.8) or nothing (p=0.2, so base-only states are visited too). At every chunk the oracle
target is recorded for several sentences at once, all at the same state:
  each present obstacle  -> its detour (zero if the base path already clears it)
  one absent obstacle    -> zero
  no sentence (null)     -> zero
so the residual has to read which object is named to know what to output.
With --residual (DAgger round), the executed chunk is base + the residual sampled for the
driving sentence (null sentence when nothing is named); labels are still the oracle detour
relative to the base chunk, now at the states the residual itself visits.
Saved per worker: feats (C,F) fp16, base (C,16,7) fp16 normalized, rows (R,) chunk/obj/kind,
targets (R,16,3) normalized xyz corrections; plus an episode log.
"""
import argparse
import json
import os
import time

import numpy as np

KIND = {"present": 0, "absent": 1, "null": 2}


def worker(args):
    wid, ids, cfg = args
    import torch
    torch.set_num_threads(1)
    import sl_env
    from controllers import BaseRunner
    from eval_sl import load_scenes
    from oracle import Oracle
    states, meta = load_scenes(cfg["scenes"])
    r = BaseRunner(cfg["ckpt"], cfg["device"], cfg["flow_steps"])
    oc = json.loads(cfg["oracle_cfg"]) if cfg["oracle_cfg"] else None
    half = (r.amax[:3] - r.amin[:3]) / 2
    res = None
    if cfg.get("residual"):
        import texts as TX
        from policy import ResidualFlow
        ck = torch.load(cfg["residual"], map_location="cpu")
        res = ResidualFlow(feat_dim=ck["feat_dim"], text_dim=768 if ck["variant"] == "text" else 0)
        res.load_state_dict(ck["model"]); res.eval().to(cfg["device"]); res_scale = ck["scale"]
        bank = TX.TextBank()
    rng = np.random.default_rng(cfg["seed"] * 1000 + wid)
    F, B, rows, T, eplog, CS, IA, IW = [], [], [], [], [], [], [], []
    envs = {}
    for i in ids:
        m = meta[i]
        key = (m["level"], m["task"], bool(m.get("extended")))
        if key not in envs:
            for v in envs.values():
                v[0].close()
            envs = {key: sl_env.make_env(m["level"], m["task"], extended=key[2])}
        env, lang, _ = envs[key]
        tid = r.task_id(lang)
        sl_env.load_state(env, states[i], m.get("bodies")); obs = sl_env.settle(env)
        present = [n for n in m["obstacles"] if n in sl_env.active_obstacles(obs)]
        absent = [n for n in sl_env.OBSTACLES if n not in present]
        drive = [present[rng.integers(len(present))]] if (present and rng.random() < 0.8) else []
        init_pos = {n: obs[n + "_pos"].copy() for n in present}
        if res is not None:
            words = TX.train_texts(drive[0]) if drive else None
            sent = words[rng.integers(len(words))] if drive else ""
            temb = torch.from_numpy(bank(sent)).unsqueeze(0).to(cfg["device"])
        oracles = {n: Oracle(oc) for n in present}
        for n, o in oracles.items():
            o.reset(env, [n])
        hits, t, success, queue, n_corr = {}, 0, False, [], 0
        while t < sl_env.MAX_STEPS:
            if not queue:
                x, f = r.chunk(obs, tid)
                xn = x.cpu().numpy()
                A = r.unnorm(xn)
                c = len(F)
                F.append(f[0].cpu().numpy().astype(np.float16)); B.append(xn.astype(np.float16)); CS.append(i)
                if cfg["save_images"]:
                    IA.append(obs["agentview_image"].copy()); IW.append(obs["robot0_eye_in_hand_image"].copy())
                exec_A = A
                if res is not None:
                    d = res.sample(f, x.unsqueeze(0), temb, steps=cfg["flow_steps"], guide=cfg["guide"])[0].cpu().numpy() / res_scale
                    exec_A = A.copy(); exec_A[:, :3] = np.clip(A[:, :3] + d * half, -1, 1)
                for n, o in oracles.items():
                    A2, prof = o.correct(env, obs, A)
                    d = ((A2[:, :3] - A[:, :3]) / half).astype(np.float32)
                    rows.append((c, sl_env.OBSTACLES.index(n), KIND["present"])); T.append(d)
                    if drive and n == drive[0] and res is None:
                        exec_A = A2
                        n_corr += int(np.abs(prof).max() > 0)
                ab = absent[rng.integers(len(absent))]
                rows.append((c, sl_env.OBSTACLES.index(ab), KIND["absent"])); T.append(np.zeros((len(A), 3), np.float32))
                rows.append((c, -1, KIND["null"])); T.append(np.zeros((len(A), 3), np.float32))
                queue = list(exec_A[: cfg["exec_steps"]])
            obs, _, done, _ = env.step(queue.pop(0).tolist())
            for n in sl_env.collided(obs, init_pos, [n for n in present if n not in hits]):
                hits[n] = t
            if done:
                success = True
                break
            t += 1
        eplog.append(dict(scene=i, level=m["level"], task=m["task"], present=present, drive=drive,
                          success=success, hits=hits, steps=t, n_corrected=n_corr,
                          drive_hit=bool(drive) and drive[0] in hits))
        if len(eplog) % 10 == 0:
            print(f"[w{wid}] {len(eplog)}/{len(ids)} episodes, {len(F)} chunks", flush=True)
    np.savez_compressed(os.path.join(cfg["out"], f"part_{wid}.npz"), feats=np.stack(F), base=np.stack(B),
                        rows=np.array(rows, np.int32), targets=np.stack(T).astype(np.float16),
                        chunk_scene=np.array(CS, np.int32))
    if cfg["save_images"]:
        np.save(os.path.join(cfg["out"], f"agent_{wid}.npy"), np.stack(IA))
        np.save(os.path.join(cfg["out"], f"wrist_{wid}.npy"), np.stack(IW))
    json.dump(eplog, open(os.path.join(cfg["out"], f"episodes_{wid}.json"), "w"))
    return wid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--exec_steps", type=int, default=8)
    ap.add_argument("--flow_steps", type=int, default=10)
    ap.add_argument("--oracle_cfg", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--residual", default="", help="DAgger: execute base + this residual instead of the oracle")
    ap.add_argument("--guide", type=float, default=1.0)
    ap.add_argument("--save_images", action="store_true", help="also store agentview/wrist frames per chunk")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from eval_sl import load_scenes
    n = len(load_scenes(a.scenes)[1])
    ids = list(range(n if not a.limit else min(n, a.limit)))
    per = int(np.ceil(len(ids) / a.workers))
    blocks = [ids[k * per:(k + 1) * per] for k in range(a.workers)]
    cfg = vars(a)
    t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for wid in pool.imap_unordered(worker, [(k, b, cfg) for k, b in enumerate(blocks) if b]):
            print(f"worker {wid} done {(time.time() - t0) / 60:.1f} min", flush=True)
    ep = sum((json.load(open(os.path.join(a.out, f))) for f in os.listdir(a.out) if f.startswith("episodes_")), [])
    dr = [e for e in ep if e["drive"]]
    nd = [e for e in ep if not e["drive"]]
    print(f"episodes {len(ep)}; driven by oracle {len(dr)}: success {np.mean([e['success'] for e in dr]):.3f}, "
          f"named obstacle hit {np.mean([e['drive_hit'] for e in dr]):.3f}; base-only {len(nd)}: success "
          f"{np.mean([e['success'] for e in nd]) if nd else float('nan'):.3f}, any hit "
          f"{np.mean([bool(e['hits']) for e in nd]) if nd else float('nan'):.3f}")


if __name__ == "__main__":
    main()
