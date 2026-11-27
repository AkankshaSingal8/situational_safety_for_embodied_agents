"""Distil pi0.5 + shield into pi0.5: fine-tune pi05_libero's action expert on rollouts where pi0.5 driven by the
lookahead shield finished the task without touching the named obstacle (meta safe_success).

    data    episodes from the same roots as train_expert.py, training objects only (held-out GSO objects -> val),
            driven by the shield, safe_success. One sample = one chunk start c: agent/wrist image and state at c,
            prompt = task instruction (pi0.5's own input, no safety sentence), target = the 10 actions actually
            executed from c (exec[c] then exec[c+1]; the last action repeats past the episode end).
    sample  p_near of the samples come from chunks within +-2 of a shield-corrected chunk (detours are ~5% of chunks)
    inputs  the exact pi05_libero policy transforms (LiberoInputs, quantile Normalize with the checkpoint's norm
            stats, resize, tokenizer, pad to 32 action dims), so the result is served by openpi serve_policy as is
    model   pi05_libero; PaliGemma frozen, the action expert + projections + time MLP trained (as train_expert.py)
    output  OUT/step_<N>/params (openpi checkpoint layout, bf16) + assets symlink -> serve with
            run_pi05_server.sh <dir under openpi-assets/checkpoints>

    python distill_pi05.py --out /data/asingal2/openpi_cache/openpi-assets/checkpoints/pi05_distill1 --steps 10000
"""
import argparse
import json
import os
import time

import numpy as np

from train_expert import episode_list, is_val_object, read_meta

ROOTS = "data_gso,data_gso_r2,data_gso_retry,data_gso_r2_retry,data_gso_dag1,data_gso_r3"
ROUNDS = "r0,r0_official,r1_*,r0_retry,dag1_*"
PI05 = "/data/asingal2/openpi_cache/openpi-assets/checkpoints/pi05_libero"
H = 10


def episodes(cache, roots, rounds):
    """(train, val) lists of (file, lang, corrected chunk ids) for driven safe-success episodes; cached as json."""
    if os.path.exists(cache):
        d = json.load(open(cache))
        return d["train"], d["val"]
    from concurrent.futures import ThreadPoolExecutor
    files = episode_list(roots, rounds)
    with ThreadPoolExecutor(32) as ex:
        metas = list(ex.map(read_meta, files))
    tr, va = [], []
    for f, m in zip(files, metas):
        if m is None or not m.get("drive"):
            continue
        # data_v2 rounds store safe_success; GSO collections store success + named_hit (collect_gso.py)
        safe = m["safe_success"] if m.get("safe_success") is not None else (m.get("success") and not m.get("named_hit"))
        if not safe:
            continue
        (va if m.get("gso") and is_val_object(m["gso"]) else tr).append([f, m["lang"]])
    json.dump(dict(train=tr, val=va), open(cache, "w"))
    return tr, va


def load_samples(args):
    """One episode file -> list of raw samples (pi0.5 client format + actions)."""
    f, lang, n, p_near, seed = args
    rng = np.random.default_rng(seed)
    try:
        with np.load(f, allow_pickle=True) as z:
            agent, wrist, state = z["agent"], z["wrist"], z["state"]
            ex, el, dt = z["exec"], z["exec_len"], z["drive_tgt"]
    except Exception:  # noqa: BLE001
        return []
    C = len(el)
    acts = np.concatenate([ex[c, : el[c]] for c in range(C)])   # executed actions in order
    start = np.concatenate([[0], np.cumsum(el)[:-1]])           # step index of each chunk start
    corr = np.where(np.abs(dt).reshape(C, -1).sum(1) > 1e-6)[0]
    near = np.unique(np.clip((corr[:, None] + np.arange(-2, 3)[None]).ravel(), 0, C - 1)) if len(corr) else corr
    out = []
    for _ in range(n):
        c = int(rng.choice(near)) if len(near) and rng.random() < p_near else int(rng.integers(C))
        idx = np.minimum(start[c] + np.arange(H), len(acts) - 1)
        out.append({"observation/image": np.ascontiguousarray(agent[c][::-1, ::-1]),
                    "observation/wrist_image": np.ascontiguousarray(wrist[c][::-1, ::-1]),
                    "observation/state": state[c].astype(np.float32),
                    "prompt": lang, "actions": acts[idx].astype(np.float32)})
    return out


def batch_iter(pool, eps, bs, per_ep, p_near, seed):
    """pool must be created before jax is imported (fork after jax start can deadlock)."""
    rng = np.random.default_rng(seed)

    def jobs():
        k = 0
        while True:
            for i in rng.permutation(len(eps)):
                k += 1
                yield (eps[i][0], eps[i][1], per_ep, p_near, seed * 1_000_003 + k)

    # bounded prefetch: imap over an endless generator reads ahead without limit and ran the host out of memory
    import collections
    gen, pending, buf = jobs(), collections.deque(), []
    depth = 2 * pool._processes
    while True:
        while len(pending) < depth:
            pending.append(pool.apply_async(load_samples, (next(gen),)))
        buf += pending.popleft().get()
        while len(buf) >= bs:
            yield buf[:bs]
            buf = buf[bs:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--per_ep", type=int, default=4)
    ap.add_argument("--p_near", type=float, default=0.5, help="share of samples near a shield-corrected chunk")
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--keep_steps", default="3000,6000,10000")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_eps", type=int, default=0, help="smoke test: use only this many training episodes")
    ap.add_argument("--cache", default="runs_v2/distill_episodes.json", help="episode list cache (built on first run)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr, va = episodes(a.cache, ROOTS, ROUNDS)
    if a.max_eps:
        tr = tr[: a.max_eps]
    print(f"safe-success driven episodes: train {len(tr)}, val {len(va)}", flush=True)
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(a.workers)                              # forks before jax is imported
    it = batch_iter(pool, tr, a.bs, a.per_ep, a.p_near, a.seed)
    vrng = np.random.default_rng(1)
    vsel = vrng.choice(len(va), min(len(va), 64), replace=False)
    vb = sum((load_samples((va[i][0], va[i][1], 2, 0.5, 7 + int(i))) for i in vsel), [])

    import dataclasses
    import pathlib

    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from flax import nnx
    from openpi.models import model as _model
    from openpi.shared import nnx_utils
    from openpi.training import checkpoints as _ckpts
    from openpi.training import config as _config
    from openpi import transforms as T

    cfg = _config.get_config("pi05_libero")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm = _ckpts.load_norm_stats(pathlib.Path(PI05) / "assets", dc.asset_id)
    tf = T.compose([T.InjectDefaultPrompt(None), *dc.data_transforms.inputs,
                    T.Normalize(norm, use_quantiles=dc.use_quantile_norm), *dc.model_transforms.inputs])
    model = cfg.model.load(_model.restore_params(pathlib.Path(PI05) / "params", dtype=jnp.float32))
    frozen_f = nnx.All(nnx.Param, nnx_utils.PathRegex(".*PaliGemma.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")))
    graphdef, frozen, train_state, rest = nnx.split(model, frozen_f, nnx.Param, ...)
    frozen = jax.tree.map(lambda x: x.astype(jnp.bfloat16), frozen)
    n_tr = sum(x.size for x in jax.tree.leaves(train_state))
    print(f"trainable {n_tr / 1e6:.0f}M, frozen {sum(x.size for x in jax.tree.leaves(frozen)) / 1e6:.0f}M", flush=True)
    sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, min(a.warmup, a.steps // 2), a.steps, a.lr * 0.1)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10))
    opt = tx.init(train_state)
    mesh = jax.sharding.Mesh(jax.devices(), ("d",))
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
    frozen, train_state, rest, opt = jax.device_put((frozen, train_state, rest, opt), rep)
    ema = jax.tree.map(lambda v: v.copy(), train_state)

    def to_batch(samples):
        xs = [tf(dict(s)) for s in samples]
        b = jax.tree.map(lambda *v: np.stack(v), *xs)
        acts = b.pop("actions")
        return jax.device_put((b, acts), shard)

    def loss_fn(ts, fz, rs, b, acts, rng, train):
        m = nnx.merge(graphdef, fz, ts, rs)
        return jnp.mean(m.compute_loss(rng, _model.Observation.from_dict(b), acts, train=train))

    @jax.jit
    def step(ts, os_, e, fz, rs, b, acts, rng):
        loss, g = jax.value_and_grad(loss_fn)(ts, fz, rs, b, acts, rng, True)
        upd, os_ = tx.update(g, os_, ts)
        ts = optax.apply_updates(ts, upd)
        e = jax.tree.map(lambda x, y: a.ema * x + (1 - a.ema) * y, e, ts)
        return ts, os_, e, loss

    @jax.jit
    def vloss(ts, fz, rs, b, acts, rng):
        return loss_fn(ts, fz, rs, b, acts, rng, False)

    vbatches = [to_batch(vb[i: i + a.bs]) for i in range(0, len(vb) - a.bs + 1, a.bs)]

    def evaluate(ts):
        return float(np.mean([float(vloss(ts, frozen, rest, b, ac, jax.random.key(k))) for k, (b, ac) in enumerate(vbatches)]))

    def save(ts, name):
        d = pathlib.Path(a.out) / name
        m = nnx.merge(graphdef, frozen, ts, rest)
        params = nnx.state(m, nnx.Param).to_pure_dict()
        params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)).astype(jnp.bfloat16), params)
        with ocp.PyTreeCheckpointer() as ck:
            ck.save((d / "params").resolve(), {"params": params}, force=True)
        if not (d / "assets").exists():
            os.symlink(os.path.join(PI05, "assets"), d / "assets")
        print(f"saved {d}", flush=True)

    keep = {int(x) for x in a.keep_steps.split(",") if x}
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    print(json.dumps(dict(step=0, val_loss=round(evaluate(train_state), 4))), flush=True)
    rng, t0, tl = jax.random.key(a.seed), time.time(), []
    for i in range(1, a.steps + 1):
        b, ac = to_batch(next(it))
        rng, k = jax.random.split(rng)
        train_state, opt, ema, loss = step(train_state, opt, ema, frozen, rest, b, ac, k)
        tl.append(loss)
        if i % 100 == 0:
            r = dict(step=i, loss=round(float(np.mean(jax.device_get(tl))), 4), sec_per_step=round((time.time() - t0) / i, 3))
            tl = []
            print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        if i % a.val_every == 0 or i == a.steps:
            r = dict(step=i, val_loss=round(evaluate(ema), 4))
            print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        if i in keep or i == a.steps:
            save(ema, f"step_{i}")
    print("done", (time.time() - t0) / 3600, "h", flush=True)


if __name__ == "__main__":
    main()
