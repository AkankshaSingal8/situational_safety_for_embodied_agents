"""Train the residual action expert (res_expert.py) on shield-labelled SafeLIBERO episodes (openpi env, JAX).

    python train_expert.py --out runs_v2/expert/ex1 --steps 30000 --bs 32

Data: every episode npz under data_v2/<suite>/{r0,r0_official} (the six SafeLIBERO obstacles) and
data_gso/<suite>/ (Google Scanned Objects, train split). Validation: the episodes of a fixed 8% of the GSO
training objects (by name hash) are never trained on; their fit measures transfer to unseen objects offline.

Rows. A batch element is one chunk of one episode with one prompt. With --mix (ex1_cleannames / ex2):
  corr  : true sentence (the audited object name), a chunk the shield corrected -> target = the verified correction
  safe  : true sentence, a chunk the shield left alone                          -> target 0 (already safe)
  wrong : names an object that is not in the scene                              -> target 0
  none  : task instruction only                                                 -> target 0
Every episode was kept only on safe success, so every target comes from a rollout that avoided the named
obstacle and finished the task. --balance_suites draws corr rows suite-first; --aug applies pi0.5's image
augmentation. Without --mix: the ex1 recipe (P_TRUE / P_ABSENT / null per row, --p_nz for true rows).
Targets are xyz residuals in env units, scaled by 1 / std(non-zero labels); base chunk and state are standardised.
"""
import argparse
import collections
import glob
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import time

import numpy as np

import texts as TX

SUITES = ["spatial", "object", "goal", "long"]
P_TRUE, P_ABSENT = 0.65, 0.15


def is_val_object(name):
    return int(hashlib.md5(("val" + name).encode()).hexdigest(), 16) % 100 < 8


def episode_list(gso_root, rounds="r0,r0_official"):
    """rounds: data_v2 sub-directories to train on (comma list; glob patterns allowed, e.g. "r0,r0_official,r1_*")."""
    out = []
    for s in SUITES:
        for rd in rounds.split(","):
            out += sorted(glob.glob(f"data_v2/{s}/{rd}/ep_*.npz"))
        for g in gso_root.split(","):   # several GSO roots allowed, e.g. "data_gso,data_gso_r2"
            out += sorted(glob.glob(f"{g}/{s}/ep_*.npz"))
    return out


def read_meta(f):
    """None for a file that cannot be read (e.g. still being written by a running collection job)."""
    try:
        with np.load(f, allow_pickle=True) as z:
            return json.loads(str(z["meta"]))
    except Exception:  # noqa: BLE001
        return None


def noun_pool():
    import gso_objects as G
    idx = G.load_index()
    # (noun, category) of every name a "wrong object" sentence may use
    pool = [(d["noun"], d["category"]) for n, d in idx.items() if d["ok"] and d["split"] == "train" and not is_val_object(n)]
    pool += [(TX.NOUN_TRAIN[o], TX.NOUN_TRAIN[o]) for o in TX.OBJ]
    return pool, idx


def sentence_for(meta, rng, idx):
    """A training wording naming this episode's driving obstacle."""
    g = meta.get("gso")
    if g:
        d = idx[g]
        # the audited name (gso_names.json), e.g. "lime green orchid pot". Not the bare category ("box", "bottle",
        # "toy"): LIBERO scenes hold other boxes and bottles, so a category alone does not say which item to avoid.
        noun = d["noun"]
    else:
        noun = TX.NOUN_TRAIN[meta["drive"][0].replace("_small_obstacle_1", "_obstacle_1")]
    return TX.T_TRAIN[rng.integers(len(TX.T_TRAIN))].format(o=noun), noun


MIX_KINDS = ("corr", "safe", "wrong", "none")


def make_rows(f, meta, rng, pool, idx, n, p_nz=0.5, mix_kind=None):
    """n (chunk, prompt, is_true, kind) rows from one episode.
    mix_kind (explicit mix, --mix): all n rows are of that kind
      corr  : true sentence, a chunk the shield corrected            -> target = the correction
      safe  : true sentence, a chunk the shield left alone            -> target = 0 (pi0.5's action was already safe)
      wrong : sentence naming something not in the scene, any chunk   -> target = 0
      none  : no sentence, any chunk                                  -> target = 0
    wrong / none take a corrected chunk half the time, so the same frame appears with and without the sentence.
    Without mix_kind: the ex1 recipe (P_TRUE / P_ABSENT / null per row, p_nz for true rows)."""
    with np.load(f, allow_pickle=True) as z:
        tgt = z["drive_tgt"]
    C = len(tgt)
    nzm = np.abs(tgt).sum((1, 2)) > 1e-6
    nz, zc = np.where(nzm)[0], np.where(~nzm)[0]
    rows = []
    for _ in range(n):
        driven = bool(meta.get("drive"))
        if mix_kind is not None:
            mk = mix_kind if (driven or mix_kind in ("wrong", "none")) else "none"
            if mk == "corr" and len(nz):
                c = int(nz[rng.integers(len(nz))])
            elif mk in ("corr", "safe"):
                mk = "safe"
                c = int(zc[rng.integers(len(zc))]) if len(zc) else int(rng.integers(C))
            else:
                c = int(nz[rng.integers(len(nz))]) if (len(nz) and rng.random() < 0.5) else int(rng.integers(C))
            kind = {"corr": "true", "safe": "true", "wrong": "absent", "none": "null"}[mk]
        else:
            u = rng.random()
            kind = ("true" if u < P_TRUE else "absent" if u < P_TRUE + P_ABSENT else "null") if driven else \
                ("absent" if u < 0.5 else "null")
            c = int(nz[rng.integers(len(nz))]) if (kind == "true" and len(nz) and rng.random() < p_nz) else int(rng.integers(C))
            mk = ("corr" if nzm[c] else "safe") if kind == "true" else ("wrong" if kind == "absent" else "none")
        if kind == "true":
            sent, noun = sentence_for(meta, rng, idx)
        elif kind == "absent":
            # a name for something NOT in the scene: never a present obstacle's name or the obstacle's category
            bad = set()
            for o in (meta.get("drive") or []) + (meta.get("others") or []):
                o = o.replace("_small_obstacle_1", "_obstacle_1")
                if o in TX.NOUN_TRAIN:
                    bad.add(TX.NOUN_TRAIN[o])
            g = meta.get("gso")
            if g:
                bad |= {idx[g]["noun"], idx[g]["category"]}
            while True:
                noun, cat = pool[rng.integers(len(pool))]
                if noun not in bad and cat not in bad:
                    break
            sent = TX.T_TRAIN[rng.integers(len(TX.T_TRAIN))].format(o=noun)
        else:
            sent = ""
        rows.append((c, sent, kind == "true", mk))
    return rows


def augment(img, rng, wrist):
    """pi0.5's training augmentation (openpi): third-person view gets a random 95% crop resized back and a
    +-5 deg rotation; both views get brightness / contrast / saturation jitter (0.3 / 0.4 / 0.5)."""
    from PIL import Image
    H, W = img.shape[:2]
    im = Image.fromarray(img)
    if not wrist:
        ch, cw = int(H * 0.95), int(W * 0.95)
        y, x = int(rng.integers(0, H - ch + 1)), int(rng.integers(0, W - cw + 1))
        im = im.crop((x, y, x + cw, y + ch)).resize((W, H), Image.BILINEAR)
        im = im.rotate(float(rng.uniform(-5, 5)), resample=Image.BILINEAR)
    a = np.asarray(im).astype(np.float32)
    a = a * rng.uniform(0.7, 1.3)                                         # brightness
    m = a.mean()
    a = (a - m) * rng.uniform(0.6, 1.4) + m                               # contrast
    g = a.mean(-1, keepdims=True)
    a = (a - g) * rng.uniform(0.5, 1.5) + g                               # saturation
    return np.clip(a, 0, 255).astype(np.uint8)


def load_batch(args):
    files, metas, seed, per_ep, pool, idx = args[:6]
    p_nz = args[6] if len(args) > 6 else 0.5
    kinds = args[7] if len(args) > 7 else [None] * len(files)
    aug = args[8] if len(args) > 8 else False
    from res_expert import prompt
    rng = np.random.default_rng(seed)
    B = dict(agent=[], wrist=[], state=[], base=[], tgt=[], prompt=[], true=[], kind=[])
    motion = bool(os.environ.get("RES_MOTION"))
    if motion:
        B["dstate"] = []
    for f, meta, mk in zip(files, metas, kinds):
        rows = make_rows(f, meta, rng, pool, idx, per_ep, p_nz, mk)
        with np.load(f, allow_pickle=True) as z:
            cs = [r[0] for r in rows]
            ag, wr = z["agent"][cs], z["wrist"][cs]
            st, ba, tg = z["state"][cs], z["base"][cs], z["drive_tgt"][cs]
            if motion:   # state change since the previous chunk (0 at the first chunk)
                ds = st - z["state"][np.maximum(np.asarray(cs) - 1, 0)]
        bd = os.environ.get("RES_BASE_DIR")
        bp = os.path.join(bd, f.replace("/", "__") + ".npy") if bd else ""
        if bd and os.path.exists(bp):   # residual on a distilled base (relabel_distill.py): base = its chunk,
            D = np.load(bp)[cs]          # same safe target action (episodes it could not read keep pi0.5's)
            nzr = (np.abs(tg).sum((1, 2)) > 1e-6)[:, None, None]
            tg = np.where(nzr, ba[..., :3] + tg - D[..., :3], 0.0).astype(np.float32)
            ba = D
        for j, (c, sent, tru, kd) in enumerate(rows):
            a_im, w_im = (augment(ag[j], rng, False), augment(wr[j], rng, True)) if aug else (ag[j], wr[j])
            B["agent"].append(a_im); B["wrist"].append(w_im); B["state"].append(st[j]); B["base"].append(ba[j])
            B["tgt"].append(tg[j] if tru else np.zeros_like(tg[j])); B["prompt"].append(prompt(meta["lang"], sent))
            B["true"].append(tru); B["kind"].append(kd)
            if motion:
                B["dstate"].append(ds[j])
    return {k: (np.stack(v) if k != "prompt" else v) for k, v in B.items()}


def corrected(meta):
    """The shield changed at least one chunk of this driven episode (its drive_tgt has non-zero rows)."""
    return bool(meta.get("drive")) and (meta.get("filter") or {}).get("n_shielded", 0) > 0


def pick(rng, n, corr, p_corr_ep, size):
    """Episode indices: with prob p_corr_ep from the corrected episodes, else uniform over all."""
    sel = rng.integers(n, size=size)
    if p_corr_ep > 0 and len(corr):
        m = rng.random(size) < p_corr_ep
        sel[m] = corr[rng.integers(len(corr), size=int(m.sum()))]
    return sel


def pick_mix(rng, metas, pools, mix, size):
    """Episode indices and row kinds for an explicit mix {kind: share}.
    corr -> corrected episodes; safe -> driven episodes (half from corrected ones, so a correction and an
    already-safe chunk of the same scene both appear); wrong / none -> any episode."""
    ks = list(mix)
    kinds = rng.choice(ks, size=size, p=np.array([mix[k] for k in ks]) / sum(mix.values()))
    sel = []
    for k in kinds:
        if k == "corr" and pools.get("corr_suite"):   # --balance_suites: pick the suite first, then a corrected episode
            cs = pools["corr_suite"]
            pl = cs[int(rng.integers(len(cs)))]
            sel.append(int(pl[rng.integers(len(pl))]))
            continue
        pl = pools["corr"] if (k == "corr" or (k == "safe" and rng.random() < 0.5)) else \
            pools["driven"] if k == "safe" else pools["all"]
        sel.append(int(pl[rng.integers(len(pl))]))
    return np.array(sel), [str(k) for k in kinds]


def mix_pools(metas, balance_suites=False):
    P = dict(corr=np.array([i for i, m in enumerate(metas) if corrected(m)], dtype=int),
             driven=np.array([i for i, m in enumerate(metas) if m.get("drive")], dtype=int),
             all=np.arange(len(metas)))
    if balance_suites:
        by = collections.defaultdict(list)
        for i in P["corr"]:
            by[metas[i].get("suite", "")].append(i)
        P["corr_suite"] = [np.array(v, dtype=int) for _, v in sorted(by.items()) if v]
    return P


def batches(files, metas, bs, per_ep, seed, pool, idx, workers=12, p_corr_ep=0.0, p_nz=0.5, mix=None, aug=False,
            balance_suites=False):
    """Endless iterator of batches built by a process pool. Call before importing jax (the pool forks).
    p_corr_ep: share of episodes drawn from the corrected ones; p_nz: chance a true row takes a corrected chunk."""
    rng = np.random.default_rng(seed)
    ne = bs // per_ep
    corr = np.array([i for i, m in enumerate(metas) if corrected(m)], dtype=int)
    pools = mix_pools(metas, balance_suites)

    def jobs():
        k = 0
        while True:
            if mix:
                sel, kinds = pick_mix(rng, metas, pools, mix, ne)
            else:
                sel, kinds = pick(rng, len(files), corr, p_corr_ep, ne), None
            k += 1
            yield ([files[i] for i in sel], [metas[i] for i in sel], seed * 100_000 + k, per_ep, pool, idx, p_nz,
                   kinds or [None] * ne, aug)

    p = mp.get_context("fork").Pool(workers)

    def gen():   # bounded prefetch: at most 2 * workers batches in flight (imap on an endless iterator queues forever)
        import collections
        q, js = collections.deque(), jobs()
        while True:
            while len(q) < 2 * workers:
                q.append(p.apply_async(load_batch, (next(js),)))
            yield q.popleft().get()

    return gen()


def flat(state):
    import jax
    return {jax.tree_util.keystr(k): np.asarray(v) for k, v in jax.tree_util.tree_flatten_with_path(state)[0]}


def unflat(state, d):
    import jax
    import jax.numpy as jnp
    leaves = jax.tree_util.tree_flatten_with_path(state)
    # LoRA adapters may be absent from a non-LoRA warm start; they keep their fresh init. Anything else must match.
    miss = [jax.tree_util.keystr(k) for k, _ in leaves[0] if jax.tree_util.keystr(k) not in d]
    assert all("lora" in m for m in miss), miss[:5]
    def get(k, v):
        x = np.asarray(d[k])
        if x.shape != v.shape and "cond_proj" in k and x.ndim == 2 and x.shape[1] == v.shape[1] and x.shape[0] < v.shape[0]:
            x = np.concatenate([x, np.zeros((v.shape[0] - x.shape[0], x.shape[1]), x.dtype)])   # --motion: new inputs start at 0
        assert x.shape == v.shape, (k, x.shape, v.shape)
        return jnp.asarray(x, v.dtype)
    return jax.tree_util.tree_unflatten(leaves[1], [get(jax.tree_util.keystr(k), v)
                                                    if jax.tree_util.keystr(k) in d else v for k, v in leaves[0]])


def compute_stats(files, metas, n=400, seed=0):
    rng = np.random.default_rng(seed)
    S, Bs, T, D = [], [], [], []
    for i in rng.choice(len(files), min(n, len(files)), replace=False):
        with np.load(files[i], allow_pickle=True) as z:
            S.append(z["state"]); Bs.append(z["base"].reshape(len(z["base"]), -1)); T.append(z["drive_tgt"])
            D.append(np.diff(z["state"], axis=0))
    S, Bs, T = np.concatenate(S), np.concatenate(Bs), np.concatenate(T)
    nz = T[np.abs(T).sum((1, 2)) > 1e-6]
    return dict(st_m=S.mean(0), st_s=S.std(0) + 1e-6, b_m=Bs.reshape(-1, 7).mean(0), b_s=Bs.reshape(-1, 7).std(0) + 1e-6, scale=float(1.0 / (nz.std() + 1e-6)), nz_frac=float(len(nz) / len(T)), ds_s=np.concatenate(D).std(0) + 1e-6)


def to_model_inputs(b, stats, tok):
    """numpy batch -> model arrays (normalised)."""
    st = (b["state"] - stats["st_m"]) / stats["st_s"]
    base = (b["base"] - stats["b_m"]) / stats["b_s"]
    H = base.shape[1]
    parts = [base, np.repeat(st[:, None], H, 1)]
    if "dstate" in b:   # --motion checkpoints
        parts.append(np.repeat((b["dstate"] / stats["ds_s"])[:, None], H, 1))
    cond = np.concatenate(parts, -1).astype(np.float32)
    toks, tmask = tok(b["prompt"], st)
    return dict(agent=b["agent"], wrist=b["wrist"], toks=toks, tmask=tmask, cond=cond,
                tgt=(b["tgt"] * stats["scale"]).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gso_root", default="data_gso")
    ap.add_argument("--p_corr_ep", type=float, default=0.0, help="share of episodes drawn from shield-corrected ones")
    ap.add_argument("--p_nz", type=float, default=0.5, help="chance a true-sentence row takes a corrected chunk")
    ap.add_argument("--val_batches", type=int, default=8)
    ap.add_argument("--mix", default="", help='explicit row mix, e.g. "corr=0.35,safe=0.30,wrong=0.15,none=0.20" '
                    "(overrides --p_corr_ep / --p_nz)")
    ap.add_argument("--rounds", default="r0,r0_official", help="data_v2 rounds, e.g. r0,r0_official,r1_*")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=32, help="global batch (split over GPUs)")
    ap.add_argument("--per_ep", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr_new", type=float, default=1e-3, help="lr of the new zero-initialised cond_proj")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_files", type=int, default=0, help="smoke test: use only this many episodes")
    ap.add_argument("--aug", action="store_true", help="pi0.5 image augmentation on training rows")
    ap.add_argument("--balance_suites", action="store_true", help="corr rows: equal share per suite")
    # optimiser; defaults reproduce ex1. pi0.5's LIBERO fine-tune (openpi config pi05_libero): batch 256, 30k steps,
    # AdamW b1 0.9 b2 0.95 eps 1e-8 wd 1e-10, clip 1.0, warmup 10k to 5e-5 then constant (--lr_end_frac 1), EMA 0.999
    ap.add_argument("--keep_steps", default="", help="comma list of steps to also save as step_<N>.pkl, e.g. 20000")
    ap.add_argument("--b2", type=float, default=0.999)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lr_end_frac", type=float, default=0.05, help="final lr / peak lr of the cosine decay (1 = constant)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the trained weights (0 = off); "
                    "validation and saved checkpoints use the EMA weights when on")
    ap.add_argument("--dup", default="", help="oversample episodes whose path contains a substring, e.g. 'dag=2' "
                    "(each matching train episode is listed that many times)")
    ap.add_argument("--gate_w", type=float, default=0.0, help="weight of the correction-needed gate BCE "
                    "(res_expert.gate_logit; res_server --gate_thr zeroes the residual when the gate is below it)")
    ap.add_argument("--init", default="", help="warm start: checkpoint .pkl whose train_state initialises the expert "
                    "(e.g. ex2 best.pkl for a DAgger fine-tune); normalisation stats are taken from it too")
    ap.add_argument("--flow_w", type=float, default=1.0, help="weight of the residual flow loss (0 = gate-only model; "
                    "best.pkl is then picked by balanced gate accuracy)")
    ap.add_argument("--gate_balance", action="store_true", help="class-balanced gate BCE")
    ap.add_argument("--lora", action="store_true", help="train rank-16 LoRA adapters on the PaliGemma backbone")
    ap.add_argument("--motion", action="store_true", help="also condition on the state change since the previous chunk "
                    "(cond_proj gets 8 zero-initialised inputs, so a warm start begins unchanged); loader workers read RES_MOTION")
    ap.add_argument("--lr_lora", type=float, default=1e-4, help="lr of the LoRA adapters")
    ap.add_argument("--base_dir", default="", help="relabel_distill.py output: train the residual on that policy's "
                    "chunks (base input and xyz target re-expressed against it); loader workers read RES_BASE_DIR")
    a = ap.parse_args()
    if a.motion:
        os.environ["RES_MOTION"] = "1"
    if a.base_dir:
        os.environ["RES_BASE_DIR"] = a.base_dir
    os.makedirs(a.out, exist_ok=True)
    files = episode_list(a.gso_root, a.rounds)
    metas = [read_meta(f) for f in files]
    if any(m is None for m in metas):
        print(f"skipping {sum(m is None for m in metas)} unreadable episode files", flush=True)
        files, metas = [f for f, m in zip(files, metas) if m is not None], [m for m in metas if m is not None]
    tr = [i for i, m in enumerate(metas) if not (m.get("gso") and is_val_object(m["gso"]))]
    va = [i for i, m in enumerate(metas) if m.get("gso") and is_val_object(m["gso"])]
    if a.max_files:
        tr = tr[:: max(1, len(tr) // a.max_files)][: a.max_files]
    n_gso = len({metas[i]["gso"] for i in tr if metas[i].get("gso")})
    print(f"episodes: train {len(tr)} ({sum(bool(metas[i].get('gso')) for i in tr)} GSO, {n_gso} GSO objects), "
          f"val {len(va)} ({len({metas[i]['gso'] for i in va})} held-out objects)", flush=True)
    if a.dup:
        for sub, w in (x.split("=") for x in a.dup.split(",")):
            extra = [i for i in tr if sub in files[i]] * (int(w) - 1)
            print(f"dup: {len(extra)} extra rows for episodes matching '{sub}'", flush=True)
            tr = tr + extra
    trf, trm = [files[i] for i in tr], [metas[i] for i in tr]
    vaf, vam = [files[i] for i in va], [metas[i] for i in va]
    stats = compute_stats(trf, trm)
    if a.init:   # keep the warm start's normalisation so its outputs keep their meaning
        with open(a.init, "rb") as f:
            init_ck = pickle.load(f)
        stats = dict(init_ck["stats"], ds_s=init_ck["stats"].get("ds_s", stats["ds_s"]))
    print("stats: scale", round(stats["scale"], 2), "non-zero label frac", round(stats["nz_frac"], 3), flush=True)
    pool, idx = noun_pool()
    print(f"corrected episodes: train {sum(map(corrected, trm))} / {len(trm)}, val {sum(map(corrected, vam))} / {len(vam)}", flush=True)
    for su in sorted({m.get("suite", "") for m in trm}):
        ms = [m for m in trm if m.get("suite", "") == su]
        print(f"  {su}: episodes {len(ms)}, corrected {sum(map(corrected, ms))}, "
              f"corrected chunks {sum((m.get('filter') or {}).get('n_shielded', 0) for m in ms if m.get('drive'))}", flush=True)
    mix = {k: float(v) for k, v in (x.split("=") for x in a.mix.split(","))} if a.mix else None
    assert mix is None or set(mix) <= set(MIX_KINDS), mix
    it_b = batches(trf, trm, a.bs, a.per_ep, a.seed, pool, idx, a.workers, a.p_corr_ep, a.p_nz, mix, a.aug,
                   a.balance_suites)   # forks: before jax is imported

    import jax
    import optax
    from flax import nnx

    import res_expert as RE

    graphdef, frozen, train_state, rest = RE.build(lora=a.lora, cond_dim=RE.COND + (RE.MOTION if a.motion else 0))
    if a.init:
        train_state = unflat(train_state, init_ck["train_state"])
        print(f"warm start from {a.init}", flush=True)
    n_tr = sum(x.size for x in jax.tree.leaves(train_state))
    n_fr = sum(x.size for x in jax.tree.leaves(frozen))
    print(f"trainable {n_tr / 1e6:.1f}M (action expert + projections + cond_proj{' + LoRA' if a.lora else ''}), frozen {n_fr / 1e6:.0f}M", flush=True)

    def label(path, _):
        if any("lora" in str(p) for p in path):
            return "lora"
        return "new" if any("cond_proj" in str(p) for p in path) else "pre"

    labels = jax.tree_util.tree_map_with_path(label, train_state)
    sched = lambda peak: optax.warmup_cosine_decay_schedule(  # noqa: E731
        0.0, peak, a.warmup, max(a.steps, a.warmup + 1), peak * a.lr_end_frac)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.multi_transform({"pre": optax.adamw(sched(a.lr), b2=a.b2, eps=1e-8, weight_decay=a.wd),
                                            "new": optax.adamw(sched(a.lr_new), b2=a.b2, eps=1e-8, weight_decay=0.0),
                                            "lora": optax.adamw(sched(a.lr_lora), b2=a.b2, eps=1e-8, weight_decay=0.0)},
                                           labels))
    opt_state = tx.init(train_state)

    ndev = jax.device_count()
    mesh = jax.sharding.Mesh(jax.devices(), ("d",))
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
    frozen, train_state, rest, opt_state = jax.device_put((frozen, train_state, rest, opt_state), rep)
    ema = jax.tree.map(lambda v: v.copy(), train_state) if a.ema else None

    @jax.jit
    def ema_update(e, ts):
        return jax.tree.map(lambda x, y: a.ema * x + (1.0 - a.ema) * y, e, ts)
    print(f"devices {ndev}, global batch {a.bs}", flush=True)

    def loss_fn(ts, fz, rs, x, rng):
        m = nnx.merge(graphdef, fz, ts, rs)
        return RE.flow_loss(m, rng, x["agent"], x["wrist"], x["toks"], x["tmask"], x["cond"], x["tgt"], gate_w=a.gate_w,
                            flow_w=a.flow_w, gate_balance=a.gate_balance)

    @jax.jit
    def step(ts, os_, fz, rs, x, rng):
        loss, g = jax.value_and_grad(loss_fn)(ts, fz, rs, x, rng)
        upd, os_ = tx.update(g, os_, ts)
        return optax.apply_updates(ts, upd), os_, loss, optax.global_norm(g)

    @jax.jit
    def predict(ts, fz, rs, x, rng):
        m = nnx.merge(graphdef, fz, ts, rs)
        return RE.sample(m, rng, x["agent"], x["wrist"], x["toks"], x["tmask"], x["cond"], steps=10, with_gate=True)

    @jax.jit
    def val_loss(ts, fz, rs, x, rng):
        return loss_fn(ts, fz, rs, x, rng)

    tok = RE.Tokenizer()
    put = lambda b: jax.device_put(to_model_inputs(b, stats, tok), shard)  # noqa: E731
    # fixed validation batches (unseen objects): true rows only for the fit, plus null/absent rows for the gate
    vrng = np.random.default_rng(1)
    vb = []
    vcorr = np.array([i for i, m in enumerate(vam) if corrected(m)], dtype=int)
    vpools = mix_pools(vam)
    for k in range(a.val_batches):
        if not vaf:
            break
        if mix:
            sel, kinds = pick_mix(vrng, vam, vpools, mix, a.bs // a.per_ep)
        else:
            sel, kinds = pick(vrng, len(vaf), vcorr, a.p_corr_ep, a.bs // a.per_ep), [None] * (a.bs // a.per_ep)
        vb.append(load_batch(([vaf[i] for i in sel], [vam[i] for i in sel], 10_000 + k, a.per_ep, pool, idx, a.p_nz,
                              kinds)))
    if vb:
        vk = collections.Counter(k for b in vb for k in b["kind"])
        print("val rows by kind:", dict(vk), flush=True)

    def evaluate(ts, it):
        if not vb:
            return {}
        L, cos, mag_t, mag_f, rp, rt, mag_s, gp = [], [], [], [], [], [], [], []
        for k, b in enumerate(vb):
            x = put(b)
            L.append(float(val_loss(ts, frozen, rest, x, jax.random.key(k))))
            p, g = predict(ts, frozen, rest, x, jax.random.key(100 + k))
            p, g = np.asarray(p), np.asarray(g)
            t = np.asarray(x["tgt"])
            gp += [(float(g[j]), bool(np.abs(t[j]).sum() > 1e-3)) for j in range(len(p))]
            for j in range(len(p)):
                if b["kind"][j] == "safe":
                    mag_s.append(np.abs(p[j]).sum(-1).max())
                if b["true"][j] and np.abs(t[j]).sum() > 1e-3:
                    cos.append(float((p[j] * t[j]).sum() / (np.linalg.norm(p[j]) * np.linalg.norm(t[j]) + 1e-8)))
                    rp.append(np.linalg.norm(p[j])); rt.append(np.linalg.norm(t[j]))
                elif not b["true"][j]:
                    mag_f.append(np.abs(p[j]).sum(-1).max())
                if b["true"][j]:
                    mag_t.append(np.abs(p[j]).sum(-1).max())
        return dict(step=it, val_loss=round(float(np.mean(L)), 4), cos_nz=round(float(np.mean(cos)), 3) if cos else None,
                    norm_ratio=round(float(np.mean(rp) / np.mean(rt)), 3) if rt else None,
                    mag_true=round(float(np.mean(mag_t)), 3), mag_null_absent=round(float(np.mean(mag_f)), 3),
                    mag_safe=round(float(np.mean(mag_s)), 3) if mag_s else None,
                    gate_p_nz=round(float(np.mean([g for g, y in gp if y])), 3) if any(y for _, y in gp) else None,
                    gate_p_zero=round(float(np.mean([g for g, y in gp if not y])), 3),
                    gate_acc=round(float(np.mean([(g >= 0.5) == y for g, y in gp])), 3),
                    gate_bal_acc=round(0.5 * float(np.mean([g >= 0.5 for g, y in gp if y] or [0])) +
                                       0.5 * float(np.mean([g < 0.5 for g, y in gp if not y] or [0])), 3))

    def save(ts, name):
        with open(os.path.join(a.out, name + ".tmp"), "wb") as f:
            pickle.dump(dict(train_state=flat(jax.device_get(ts)), stats=stats, args=vars(a)), f)
        os.replace(os.path.join(a.out, name + ".tmp"), os.path.join(a.out, name))

    json.dump(dict(args=vars(a), stats={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stats.items()}),
              open(os.path.join(a.out, "config.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    rng = jax.random.key(a.seed)
    t0, tl = time.time(), []
    best = -1e9
    for it in range(1, a.steps + 1):
        x = put(next(it_b))
        rng, k = jax.random.split(rng)
        train_state, opt_state, loss, gn = step(train_state, opt_state, frozen, rest, x, k)
        if ema is not None:
            ema = ema_update(ema, train_state)
        ev = ema if ema is not None else train_state     # weights used for validation and checkpoints
        tl.append(loss)
        if it % 100 == 0:
            r = dict(step=it, loss=round(float(np.mean(jax.device_get(tl))), 4), gnorm=round(float(gn), 3),
                     sec_per_step=round((time.time() - t0) / it, 3),
                     peak_gb=round(max((d.memory_stats() or {}).get("peak_bytes_in_use", 0) for d in jax.devices()) / 2**30, 1))
            tl = []
            print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        if it % a.val_every == 0 or it == a.steps:
            r = evaluate(ev, it)
            print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
            score = r.get("gate_bal_acc") if not a.flow_w else r.get("cos_nz")
            if score is not None and score > best:
                best = score
                save(ev, "best.pkl")
        if it % a.save_every == 0 or it == a.steps:
            save(ev, "last.pkl")
        if a.keep_steps and it in {int(x) for x in a.keep_steps.split(",")}:
            save(ev, f"step_{it}.pkl")
    print("done", (time.time() - t0) / 3600, "h", flush=True)


if __name__ == "__main__":
    main()
