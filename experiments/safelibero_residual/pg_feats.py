"""Cache frozen pi05_base PaliGemma features for the v2 residual data (openpi env).

    # 1. fit the PCA once (random chunks from every suite)
    python pg_feats.py --fit_pca --out data_v2/feats_pg
    # 2. per suite
    python pg_feats.py --suite spatial --rounds r0 --out data_v2/feats_pg

The fused features depend on the sentence, so each chunk is encoded once per prompt variant ("row"):
  driven chunk : kind 0 (true: a training wording naming the driving obstacle)
                 + one of kind 1 (absent: names an obstacle type not in the scene) / kind 2 (null: instruction only)
  other chunk  : one row, kind 1 or 2
Outputs in <out>/:
  <s>_pg.npy    (R,128,K) fp16   fused Gemma tokens per row (backbone C)
  <s>_pgm.npy   (R,48) bool      valid text tokens per row
  <s>_sig.npy   (N,80,K) fp16    SigLIP-So400m tokens per chunk (backbone B; sentence-independent)
  <s>_rows.npz  chunk (R,), kind (R,), obj (R,) index into texts.OBJ or -1, wording (R,)
  <s>_index.npz state, base, drive_tgt, ep, chunk, eps_meta (same layout as cache_feats_v2.py)
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import texts as TX  # noqa: E402
from pg_backbone import MAX_TXT, N_IMG, Backbone, prompt  # noqa: E402

K = 384
SUITES = ["spatial", "object", "goal", "long"]


def canon(n):
    return n.replace("_small_obstacle_1", "_obstacle_1")


def files_of(s, rounds):
    return sorted(sum((glob.glob(f"data_v2/{s}/{r}/ep_*.npz") for r in rounds), []))


def ep_rows(meta, n, rng):
    """Rows (chunk, kind, obj, wording, prompt) for one episode of n chunks."""
    pres = {canon(x) for x in meta["drive"] + meta["others"]}
    absent = [i for i, o in enumerate(TX.OBJ) if o not in pres]
    drv = TX.OBJ.index(canon(meta["drive"][0])) if meta["drive"] else -1
    rows = []
    for c in range(n):
        if drv >= 0:
            w = int(rng.integers(len(TX.T_TRAIN)))
            rows.append((c, 0, drv, w, prompt(meta["lang"], TX.train_texts(TX.OBJ[drv])[w])))
        if absent and rng.random() < 0.5:
            o = absent[rng.integers(len(absent))]
            w = int(rng.integers(len(TX.T_TRAIN)))
            rows.append((c, 1, o, w, prompt(meta["lang"], TX.train_texts(TX.OBJ[o])[w])))
        else:
            rows.append((c, 2, -1, -1, prompt(meta["lang"])))
    return rows


def fit_pca(bb, out, n_eps=60, seed=0):
    rng = np.random.default_rng(seed)
    PG, SG = [], []
    for s in SUITES:
        fs = files_of(s, ["r0"])
        for f in rng.choice(fs, min(n_eps, len(fs)), replace=False):
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            n = len(z["state"])
            rows = ep_rows(meta, n, rng)
            sel = rng.choice(len(rows), min(8, len(rows)), replace=False)
            c = np.array([rows[i][0] for i in sel])
            pg, sig, m = bb.raw(z["agent"][c], z["wrist"][c], [rows[i][4] for i in sel])
            valid = np.concatenate([np.ones((len(sel), N_IMG), bool), m], 1)
            PG.append(pg[valid]); SG.append(sig.reshape(-1, sig.shape[-1]))
        print(s, "pca samples", sum(len(x) for x in PG), flush=True)
    res = {}
    for name, X in (("pg", np.concatenate(PG)), ("sig", np.concatenate(SG))):
        X = X.astype(np.float64)
        mu = X.mean(0)
        U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
        ev = S ** 2 / (S ** 2).sum()
        res[f"{name}_mean"] = mu.astype(np.float32)
        res[f"{name}_proj"] = Vt[:K].T.astype(np.float32)
        print(f"{name}: {len(X)} tokens, dim {X.shape[1]}, variance kept by {K} comps {ev[:K].sum():.4f}", flush=True)
    np.savez(f"{out}/pca.npz", **res)


def cache_suite(bb, s, rounds, out, bs=48, seed=0, name=None):
    files = files_of(s, rounds)
    s_out = name or s
    rng = np.random.default_rng(seed * 7919 + SUITES.index(s))
    metas, lens = [], []
    for f in files:
        z = np.load(f)
        metas.append(json.loads(str(z["meta"])))
        lens.append(len(z["exec_len"]))
    N = int(sum(lens))
    rows_all, off = [], 0
    for e, (m, n) in enumerate(zip(metas, lens)):
        rows_all.append([(off + c, k, o, w, p) for c, k, o, w, p in ep_rows(m, n, rng)])
        off += n
    R = sum(len(r) for r in rows_all)
    print(f"{s}: {len(files)} eps, {N} chunks, {R} rows", flush=True)
    # resume: rows are deterministic (seeded rng), so a checkpoint only needs the episodes done and the masks
    ck_path = f"{out}/{s_out}_ckpt.npz"
    e_done = 0
    if os.path.exists(ck_path) and os.path.exists(f"{out}/{s_out}_pg.npy"):
        ck = np.load(ck_path)
        assert int(ck["R"]) == R and int(ck["N"]) == N, "checkpoint is from a different row layout"
        e_done = int(ck["e_done"])
        PG = np.lib.format.open_memmap(f"{out}/{s_out}_pg.npy", "r+")
        SIG = np.lib.format.open_memmap(f"{out}/{s_out}_sig.npy", "r+")
        PGM = ck["pgm"].copy()
        print(f"{s}: resuming after {e_done} eps", flush=True)
    else:
        PG = np.lib.format.open_memmap(f"{out}/{s_out}_pg.npy", "w+", np.float16, (R, N_IMG + MAX_TXT, K))
        PGM = np.zeros((R, MAX_TXT), bool)
        SIG = np.lib.format.open_memmap(f"{out}/{s_out}_sig.npy", "w+", np.float16, (N, N_IMG, K))
    ST, B, T, EP, CH, M = [], [], [], [], [], []
    offs = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    r0, t0 = 0, time.time()
    buf = []  # (row id, agent, wrist, prompt, chunk id, first row of chunk?)

    def flush():
        nonlocal buf
        if not buf:
            return
        pg, sig, msk = bb(np.stack([b[1] for b in buf]), np.stack([b[2] for b in buf]), [b[3] for b in buf])
        for j, b in enumerate(buf):
            PG[b[0]] = pg[j]; PGM[b[0]] = msk[j]
            if b[5]:
                SIG[b[4]] = sig[j]
        buf = []

    for e, f in enumerate(files):
        z = np.load(f)
        meta = metas[e]
        n = lens[e]
        seen = set()
        if e < e_done:                      # already encoded in an earlier run
            r0 += len(rows_all[e])
        else:
            A, W = z["agent"], z["wrist"]
        for gc, k, o, w, p in (rows_all[e] if e >= e_done else []):
            lc = gc - offs[e]
            buf.append((r0, A[lc], W[lc], p, gc, gc not in seen))
            seen.add(gc)
            r0 += 1
            if len(buf) == bs:
                flush()
        ST.append(z["state"]); B.append(z["base"]); T.append(z["drive_tgt"])
        EP.append(np.full(n, e, np.int32)); CH.append(np.arange(n, dtype=np.int32))
        M.append(json.dumps({k: meta[k] for k in ("scene", "layout", "level", "task", "lang", "drive", "others", "hits")}
                            | {"suite": s, "file": os.path.relpath(f)}))
        if e % 100 == 0 and e >= e_done:
            flush(); PG.flush(); SIG.flush()
            np.savez(ck_path + ".tmp.npz", e_done=e + 1, R=R, N=N, pgm=PGM)
            os.replace(ck_path + ".tmp.npz", ck_path)
            el = time.time() - t0
            print(f"{s}: {e}/{len(files)} eps, {r0}/{R} rows, {r0 / max(el, 1):.1f} rows/s", flush=True)
    flush()
    PG.flush(); SIG.flush()
    np.save(f"{out}/{s_out}_pgm.npy", PGM)
    rr = [r for rows in rows_all for r in rows]
    np.savez(f"{out}/{s_out}_rows.npz", chunk=np.array([r[0] for r in rr], np.int64), kind=np.array([r[1] for r in rr], np.int8),
             obj=np.array([r[2] for r in rr], np.int8), wording=np.array([r[3] for r in rr], np.int8))
    np.savez(f"{out}/{s_out}_index.npz", state=np.concatenate(ST), base=np.concatenate(B), drive_tgt=np.concatenate(T),
             ep=np.concatenate(EP), chunk=np.concatenate(CH), eps_meta=np.array(M))
    if os.path.exists(ck_path):
        os.remove(ck_path)
    print(f"{s}: done, {R} rows in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_v2/feats_pg")
    ap.add_argument("--fit_pca", action="store_true")
    ap.add_argument("--suite", default="")
    ap.add_argument("--rounds", default="r0")
    ap.add_argument("--name", default="", help="output prefix (default: suite)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.fit_pca:
        fit_pca(Backbone(), a.out)
        return
    bb = Backbone(pca=f"{a.out}/pca.npz")
    cache_suite(bb, a.suite, a.rounds.split(","), a.out, name=a.name or None)


if __name__ == "__main__":
    main()
