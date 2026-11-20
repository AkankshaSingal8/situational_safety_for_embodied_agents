"""Train the v2 residual flow jointly on all suites from cached SigLIP features.

    python train_v2.py --feats data_v2/feats_siglip --suites spatial,object,goal,long --out runs_v2/train/joint_r0

Rows per batch: 50% present (sentence names the episode's driving obstacle, target = verified drive_tgt),
25% absent (sentence names an obstacle not in the scene, target 0), 25% null (no sentence, target 0).
Present rows come only from driven episodes; absent and null rows from every chunk. Sentences are drawn
from the 8 training wordings at every step. Rows naming a present but non-driving obstacle are never used
(their labels were not rollout-verified). All data is training data (official episodes 0-39); there is no
validation split, so the fit printed at the end is training fit.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

import texts as TX
from policy_v2 import ResidualFlowV2


def canon(n):
    return n.replace("_small_obstacle_1", "_obstacle_1")


def load(feats, suites):
    A, W, ST, B, T, DR, PRES, LANG = [], [], [], [], [], [], [], []
    for s in suites:
        z = np.load(f"{feats}/{s}_index.npz")
        metas = [json.loads(m) for m in z["eps_meta"]]
        ep = z["ep"]
        A.append(np.load(f"{feats}/{s}_agent.npy")); W.append(np.load(f"{feats}/{s}_wrist.npy"))
        ST.append(z["state"]); B.append(z["base"]); T.append(z["drive_tgt"])
        dr = np.array([TX.OBJ.index(canon(m["drive"][0])) if m["drive"] else -1 for m in metas])
        pres = np.zeros((len(metas), len(TX.OBJ)), bool)
        for i, m in enumerate(metas):
            for n in m["drive"] + m["others"]:
                pres[i, TX.OBJ.index(canon(n))] = True
        DR.append(dr[ep]); PRES.append(pres[ep]); LANG += [metas[e]["lang"] for e in ep]
        print(f"{s}: {len(metas)} eps ({int((dr >= 0).sum())} driven), {len(ep)} chunks", flush=True)
    return [np.concatenate(x) for x in (A, W, ST, B, T, DR, PRES)] + [LANG]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="data_v2/feats_siglip")
    ap.add_argument("--suites", default="spatial,object,goal,long")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nz_frac", type=float, default=0.0,
                    help="fraction of present rows drawn from chunks with a non-zero label (0 = uniform over driven chunks)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda"
    A, W, ST, B, T, DR, PRES, LANG = load(a.feats, a.suites.split(","))
    N = len(A)
    # text
    tb = np.load(f"{a.feats}/text_bank.npz", allow_pickle=True)
    tix = {t: i for i, t in enumerate(tb["texts"].tolist())}
    temb = tb["emb"].astype(np.float32)
    sent = np.stack([np.stack([temb[tix[s]] for s in TX.train_texts(o)]) for o in TX.OBJ])  # (6,8,768)
    langs = sorted(set(LANG))
    lang_id = np.array([langs.index(l) for l in LANG])
    instr = np.stack([temb[tix[l]] for l in langs])
    # normalisation
    st_m, st_s = ST.mean(0), ST.std(0) + 1e-6
    bf = B.reshape(N, -1)
    b_m, b_s = bf.mean(0), bf.std(0) + 1e-6
    drv = DR >= 0
    nz = np.abs(T[drv]).max(axis=(1, 2)) > 0
    scale = float(1.0 / (T[drv][nz].std() + 1e-6))
    print(f"chunks {N}, driven {int(drv.sum())} ({nz.mean():.3f} non-zero target), target scale {scale:.2f}, "
          f"{len(langs)} instructions", flush=True)
    feat_gb = (A.nbytes + W.nbytes) / 1e9
    fdev = dev if feat_gb < 32 else "cpu"
    print(f"image features {feat_gb:.1f} GB on {fdev}", flush=True)
    g = lambda x, d=dev: torch.from_numpy(x).to(d)
    Ag, Wg = g(A, fdev), g(W, fdev)
    if fdev == "cpu":
        Ag, Wg = Ag.pin_memory(), Wg.pin_memory()
    del A, W
    STg = g(((ST - st_m) / st_s).astype(np.float32))
    Bg = g(((bf - b_m) / b_s).astype(np.float32).reshape(B.shape))
    Tg = g((T * scale).astype(np.float32))
    DRg, ABSg = g(DR.astype(np.int64)), g((~PRES).astype(np.float32))
    SENTg, INSTRg, LIDg = g(sent), g(instr), g(lang_id.astype(np.int64))
    drv_idx = np.flatnonzero(drv)
    nz_idx = drv_idx[np.abs(T[drv_idx]).max(axis=(1, 2)) > 0]

    def present_idx(n):
        k = int(round(n * a.nz_frac))
        return np.concatenate([rng.choice(nz_idx, k), rng.choice(drv_idx, n - k)])
    nW = sent.shape[1]

    def batch(idx_np, kind, random_wording=True):
        """kind: 0 present, 1 absent, 2 null (one kind per call)."""
        idx = torch.from_numpy(idx_np).to(dev)
        n = len(idx)
        w = torch.randint(0, nW, (n,), device=dev) if random_wording else torch.zeros(n, dtype=torch.long, device=dev)
        has = torch.ones(n, device=dev)
        tgt = torch.zeros(n, Tg.shape[1], 3, device=dev)
        if kind == 0:
            obj = DRg[idx]; tgt = Tg[idx]
        elif kind == 1:
            m = ABSg[idx]
            ok = m.sum(1) > 0
            obj = torch.multinomial(m + (~ok).float().unsqueeze(1), 1)[:, 0]
            has = ok.float()
        else:
            obj = torch.zeros(n, dtype=torch.long, device=dev); has = torch.zeros(n, device=dev)
        ii = idx.to(fdev)
        kw = dict(agent=Ag[ii].to(dev, non_blocking=True), wrist=Wg[ii].to(dev, non_blocking=True),
                  sent=SENTg[obj, w], has_sent=has, instr=INSTRg[LIDg[idx]], state_n=STg[idx], base_n=Bg[idx])
        return kw, tgt

    model = ResidualFlowV2(horizon=B.shape[1], layers=a.layers).to(dev)
    ema = copy.deepcopy(model).eval()
    print(f"trainable params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + np.cos(np.pi * s / a.steps)))
    nb = [a.bs // 2, a.bs // 4, a.bs - a.bs // 2 - a.bs // 4]
    stats = dict(st_m=st_m, st_s=st_s, b_m=b_m, b_s=b_s, scale=scale)

    def save(name):
        torch.save(dict(model=ema.state_dict(), cfg=dict(horizon=int(B.shape[1]), layers=a.layers),
                        stats={k: (np.asarray(v).tolist()) for k, v in stats.items()}, suites=a.suites,
                        text_bank=os.path.abspath(f"{a.feats}/text_bank.npz")), os.path.join(a.out, name))

    log = open(os.path.join(a.out, "train_log.jsonl"), "w")
    t0, run = time.time(), None
    for step in range(1, a.steps + 1):
        loss = 0.0
        for k, (idx_np) in enumerate([present_idx(nb[0]), rng.integers(0, N, nb[1]), rng.integers(0, N, nb[2])]):
            kw, tgt = batch(idx_np, k)
            loss = loss + model.loss(tgt, **kw) * len(idx_np) / a.bs
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.999).add_(pm.detach(), alpha=0.001)
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 1000 == 0:
            rec = dict(step=step, loss=round(run, 5), min=round((time.time() - t0) / 60, 1))
            print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if step % 10000 == 0:
            save("residual.pt")
    save("residual.pt")
    # training fit (no validation split by design)
    frng = np.random.default_rng(1)
    nzi = drv_idx[np.abs(T[drv_idx]).max(axis=(1, 2)) > 0]
    res = {}
    for name, idx_np, kind in [("present_nz", frng.choice(nzi, 2000), 0), ("absent", frng.integers(0, N, 2000), 1),
                               ("null", frng.integers(0, N, 2000), 2)]:
        P, G = [], []
        for i in range(0, len(idx_np), 250):
            kw, tgt = batch(idx_np[i:i + 250], kind, random_wording=False)
            P.append(ema.sample(n=4, **kw).cpu().numpy()); G.append(tgt.cpu().numpy())
        p, t = np.concatenate(P).reshape(len(idx_np), -1), np.concatenate(G).reshape(len(idx_np), -1)
        if kind == 0:
            pn, tn = np.linalg.norm(p, axis=1), np.linalg.norm(t, axis=1)
            res[name] = dict(R2=1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum(), ratio=pn.mean() / tn.mean(),
                             cos=((p * t).sum(1) / (pn * tn + 1e-9)).mean(), tgt_norm=tn.mean())
        else:
            res[name] = dict(mag=np.linalg.norm(p, axis=1).mean())
        res[name] = {k: round(float(v), 3) for k, v in res[name].items()}
        print(name, res[name], flush=True)
    json.dump(res, open(os.path.join(a.out, "fit.json"), "w"))


if __name__ == "__main__":
    main()
