"""Run a distilled pi0.5 on every stored chunk of the residual training episodes, so a residual can be trained on top
of that policy instead of pi05_libero.

    for each episode file f (same list as train_expert.py --gso_root/--rounds), for each chunk c:
        D[c] = distilled policy chunk (H,7) from agent/wrist/state at c and the task language (pi0.5 client format)
    saved as OUT/<f with / -> __>.npy (float32, (C,H,7)); train_expert.py --base_dir OUT then uses
        base  = D[c]                                     (the residual's base-chunk input)
        tgt   = A + drive_tgt - D on xyz where the shield changed the chunk, else 0
    where A is the stored pi0.5 chunk, so the safe target action (what was executed / planned) stays the same.

    python relabel_distill.py --ckpt pi05_distill1/step_6000 --out /data/asingal2/relabel/d6000 --shard 0/4
"""
import argparse
import json
import os
import time

import numpy as np

from train_expert import episode_list

ROOTS = "data_gso,data_gso_r2,data_gso_retry,data_gso_r2_retry,data_gso_dag1,data_gso_r3"
ROUNDS = "r0,r0_official,r1_*,r0_retry,dag1_*"
CK = "/data/asingal2/openpi_cache/openpi-assets/checkpoints"


def out_path(out, f):
    return os.path.join(out, f.replace("/", "__") + ".npy")


def load(args):
    f, out = args
    if os.path.exists(out_path(out, f)):
        return f, None
    try:
        with np.load(f, allow_pickle=True) as z:
            lang = json.loads(str(z["meta"]))["lang"]
            return f, (z["agent"][:, ::-1, ::-1].copy(), z["wrist"][:, ::-1, ::-1].copy(), z["state"].astype(np.float32), lang, z["base"].astype(np.float32))
    except Exception as e:  # noqa: BLE001
        print("bad", f, e, flush=True)
        return f, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="dir under openpi-assets/checkpoints, e.g. pi05_distill1/step_6000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list", default="runs_v2/relabel_files.json", help="episode list cache")
    ap.add_argument("--max_files", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(a.list):
        files = json.load(open(a.list))
    else:
        files = episode_list(ROOTS, ROUNDS)
        json.dump(files, open(a.list, "w"))
    k, n = map(int, a.shard.split("/"))
    files = files[k::n][: a.max_files or None]
    print(f"shard {a.shard}: {len(files)} episodes", flush=True)
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(a.workers)   # before jax starts

    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.policies import policy_config
    from openpi.training import config as _config
    pol = policy_config.create_trained_policy(_config.get_config("pi05_libero"), os.path.join(CK, a.ckpt))
    sample = pol._sample_actions   # openpi module_jit: weights are arguments (jax.jit on the bound method baked
                                   # them in as constants and took ~75 GB of host memory to compile)
    rng = jax.random.key(k)

    def run(items):
        """items: list of (key, sample dict) -> {key: (H,7) actions}"""
        nonlocal rng
        xs = [pol._input_transform(dict(s)) for _, s in items]
        pad = a.bs - len(xs)
        xs += [xs[-1]] * pad                                   # fixed batch shape -> one compile
        b = jax.tree.map(lambda *v: jnp.asarray(np.stack(v)), *xs)
        rng, r = jax.random.split(rng)
        acts = np.asarray(sample(r, _model.Observation.from_dict(b)))
        st = np.asarray(b["state"])
        res = {}
        for i, (key, _) in enumerate(items):
            o = pol._output_transform({"state": st[i], "actions": acts[i]})
            res[key] = np.asarray(o["actions"], np.float32)
        return res

    import collections
    import resource

    def rss():
        return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB"

    print("policy loaded, peak rss", rss(), flush=True)

    def loaded():
        """bounded prefetch: at most 2 x workers episodes read ahead (unbounded imap ran a host out of memory)"""
        pend, it = collections.deque(), iter(files)
        for f in it:
            pend.append(pool.apply_async(load, ((f, a.out),)))
            if len(pend) >= 2 * a.workers:
                yield pend.popleft().get()
        while pend:
            yield pend.popleft().get()

    t0, done, buf, eps, base, dif = time.time(), 0, [], {}, {}, []
    for f, d in loaded():
        if d is None:
            continue
        ag, wr, st, lang, base[f] = d
        eps[f] = [None] * len(st)
        for c in range(len(st)):
            buf.append(((f, c), {"observation/image": ag[c], "observation/wrist_image": wr[c],
                                 "observation/state": st[c], "prompt": lang}))
        while len(buf) >= a.bs:
            for (g, c), v in run(buf[: a.bs]).items():
                eps[g][c] = v
            buf = buf[a.bs:]
            for g in [g for g, v in eps.items() if all(x is not None for x in v)]:
                D = np.stack(eps.pop(g))
                np.save(out_path(a.out, g), D)
                A = base.pop(g)   # sanity: distance of the distilled chunk from the stored pi0.5 chunk
                dif.append([np.abs(D[..., :3] - A[..., :3]).mean(), np.abs(A[..., :3]).mean(), np.abs(D[..., 6] - A[..., 6]).mean()])
                done += 1
                if done in (5, 20):
                    print("mean |D-A| xyz, |A| xyz, |D-A| grip", np.round(np.mean(dif, 0), 3), flush=True)
                if done % 200 == 0:
                    print(f"{done}/{len(files)} episodes, {(time.time() - t0) / done:.2f} s/ep, peak rss {rss()}", flush=True)
    while buf:
        for (g, c), v in run(buf[: a.bs]).items():
            eps[g][c] = v
        buf = buf[a.bs:]
    for g, v in eps.items():
        np.save(out_path(a.out, g), np.stack(v))
        done += 1
    print("peak rss", rss(), "mean |D-A| xyz, |A| xyz, |D-A| grip", np.round(np.mean(dif, 0), 3) if dif else None)
    print("done", done, "episodes", round((time.time() - t0) / 3600, 2), "h", flush=True)


if __name__ == "__main__":
    main()
