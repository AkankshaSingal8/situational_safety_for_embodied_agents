"""Train the residual flow head on frozen backbone features, jointly on all suites.

    python train_pg.py --backbone pg  --out runs_v2/train/pg_sel --holdout     # backbone selection run
    python train_pg.py --backbone pg  --out runs_v2/train/pg_all               # final: every chunk

Backbones (all frozen, features cached):
  pg      pi05_base PaliGemma, image + "<instruction>. <sentence>" fused by Gemma   (data_v2/feats_pg/<s>_pg.npy)
  sig     pi05_base SigLIP-So400m image tokens + SigLIP-base text embeddings        (data_v2/feats_pg/<s>_sig.npy)
  siglip  SigLIP-base image tokens + SigLIP-base text embeddings                    (data_v2/feats_siglip)
Rows (chunk, kind, obstacle, wording) come from data_v2/feats_pg/<s>_rows.npz for every backbone, so the three see
identical training signal. kind 0 = sentence names the driving obstacle (target = verified label),
kind 1 = names an absent obstacle (target 0), kind 2 = no sentence (target 0).
Batch: 50% kind 0 (nz_frac of those from chunks with a non-zero label), 25% kind 1, 25% kind 2.

--holdout: episodes from layouts 35-39 and every episode with a milk carton in it are held out, and no training
row names the milk carton. Validation reports the fit on (a) new layouts, seen obstacles and (b) the unseen milk
carton, plus the output magnitude for absent / null sentences. Used only to choose the backbone.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

import texts as TX
from policy_pg import ResidualFlowTok

SUITES = ["spatial", "object", "goal", "long"]
HELD_OBJ = "milk_obstacle_1"


def canon(n):
    return n.replace("_small_obstacle_1", "_obstacle_1")


def load(a):
    P = dict(tok=[], mask=[], st=[], base=[], tgt=[], chunk=[], kind=[], obj=[], word=[], lang=[], layout=[],
             has_held=[], pres=[], drv=[], bad=[])
    noff, roff = 0, 0
    for s in a.suites.split(","):
        z = np.load(f"{a.pg}/{s}_index.npz")
        rows = np.load(f"{a.pg}/{s}_rows.npz")
        metas = [json.loads(m) for m in z["eps_meta"]]
        ep = z["ep"]
        n = len(ep)
        if a.backbone == "pg":
            tok = np.load(f"{a.pg}/{s}_pg.npy", mmap_mode="r")
            P["tok"].append(np.ascontiguousarray(tok)); P["mask"].append(np.load(f"{a.pg}/{s}_pgm.npy"))
        elif a.backbone == "sig":
            P["tok"].append(np.ascontiguousarray(np.load(f"{a.pg}/{s}_sig.npy", mmap_mode="r")))
        else:
            ag = np.load(f"{a.siglip}/{s}_agent.npy", mmap_mode="r")
            wr = np.load(f"{a.siglip}/{s}_wrist.npy", mmap_mode="r")
            zs = np.load(f"{a.siglip}/{s}_index.npz")
            assert [json.loads(m)["file"] for m in zs["eps_meta"]] == [m["file"] for m in metas], "episode order differs"
            P["tok"].append(np.concatenate([ag, wr], 1))
        P["st"].append(z["state"]); P["base"].append(z["base"]); P["tgt"].append(z["drive_tgt"])
        P["chunk"].append(rows["chunk"] + noff); P["kind"].append(rows["kind"]); P["obj"].append(rows["obj"])
        P["word"].append(rows["wording"])
        lay = np.array([m["layout"] for m in metas]); P["layout"].append(lay[ep])
        held = np.array([any(canon(x) == HELD_OBJ for x in m["drive"] + m["others"]) for m in metas]); P["has_held"].append(held[ep])
        drv = np.array([TX.OBJ.index(canon(m["drive"][0])) if m["drive"] else -1 for m in metas]); P["drv"].append(drv[ep])
        # audit fix: a kept driven episode that hit any obstacle (unnamed ones count for CAR) is not a safe label
        bad = np.array([bool(m["drive"]) and bool(m.get("hits")) for m in metas]); P["bad"].append(bad[ep])
        P["lang"] += [metas[e]["lang"] for e in ep]
        print(f"{s}: {len(metas)} eps ({int(bad.sum())} dropped: driven + obstacle hit), {n} chunks, {len(rows['chunk'])} rows", flush=True)
        noff += n; roff += len(rows["chunk"])
    out = {k: (np.concatenate(v) if k != "lang" and v else v) for k, v in P.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["pg", "sig", "siglip"])
    ap.add_argument("--pg", default="data_v2/feats_pg")
    ap.add_argument("--siglip", default="data_v2/feats_siglip")
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--nz_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"))
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda"
    D = load(a)
    N, R = len(D["st"]), len(D["chunk"])
    chunk, kind, obj, word = D["chunk"], D["kind"].astype(np.int64), D["obj"].astype(np.int64), D["word"].astype(np.int64)
    T = D["tgt"]
    # ---- split ----
    held_chunk = np.zeros(N, bool)
    if a.holdout:
        held_chunk = (D["layout"] >= 35) | D["has_held"]
    hobj = TX.OBJ.index(HELD_OBJ)
    ok_row = ~D["bad"][chunk]
    row_train = ~held_chunk[chunk] & ok_row
    if a.holdout:
        row_train &= obj != hobj
    print(f"rows {R}, train rows {int(row_train.sum())}, held chunks {int(held_chunk.sum())}/{N}", flush=True)
    # ---- text (A/B only) ----
    text_dim = 0
    if a.backbone != "pg":
        tb = np.load(f"{a.siglip}/text_bank.npz", allow_pickle=True)
        tix = {t: i for i, t in enumerate(tb["texts"].tolist())}
        temb = tb["emb"].astype(np.float32)
        text_dim = temb.shape[1]
        sent = np.stack([np.stack([temb[tix[s]] for s in TX.train_texts(o)]) for o in TX.OBJ])  # (6,8,S)
        langs = sorted(set(D["lang"]))
        lang_id = np.array([langs.index(l) for l in D["lang"]])
        instr = np.stack([temb[tix[l]] for l in langs])
    # ---- normalisation ----
    tr_chunks = ~held_chunk & ~D["bad"]
    st_m, st_s = D["st"][tr_chunks].mean(0), D["st"][tr_chunks].std(0) + 1e-6
    bf = D["base"].reshape(N, -1)
    b_m, b_s = bf[tr_chunks].mean(0), bf[tr_chunks].std(0) + 1e-6
    k0 = row_train & (kind == 0)
    tnz = np.abs(T[chunk[k0]]).max(axis=(1, 2)) > 0
    scale = float(1.0 / (T[chunk[k0]][tnz].std() + 1e-6))
    print(f"kind0 train rows {int(k0.sum())} ({tnz.mean():.3f} non-zero), scale {scale:.2f}", flush=True)
    # ---- tensors ----
    tok = D["tok"]
    feat_gb = tok.nbytes / 1e9
    fdev = dev if feat_gb < 30 else "cpu"
    print(f"tokens {tok.shape} {feat_gb:.1f} GB on {fdev}", flush=True)
    g = lambda x, d=dev: torch.from_numpy(np.ascontiguousarray(x)).to(d)  # noqa: E731
    TOK = g(tok, fdev)
    if fdev == "cpu":
        TOK = TOK.pin_memory()
    MASK = g(D["mask"], fdev) if a.backbone == "pg" else None
    del tok, D["tok"]
    STg = g(((D["st"] - st_m) / st_s).astype(np.float32))
    Bg = g(((bf - b_m) / b_s).astype(np.float32).reshape(D["base"].shape))
    Tg = g((T * scale).astype(np.float32))
    CHg, KINDg, OBJg, WORDg = g(chunk), g(kind), g(obj), g(word)
    if text_dim:
        SENTg, INSTRg, LIDg = g(sent), g(instr), g(lang_id.astype(np.int64))
    per_row = a.backbone == "pg"

    def batch(ridx, wording=None):
        r = torch.from_numpy(ridx).to(dev)
        c = CHg[r]
        k = KINDg[r]
        tgt = Tg[c] * (k == 0).float().view(-1, 1, 1)
        src = (r if per_row else c).to(fdev)
        tk = TOK[src].to(dev, non_blocking=True)
        if per_row:
            tm = MASK[src].to(dev, non_blocking=True)
            tm = torch.cat([torch.ones(len(r), tk.shape[1] - tm.shape[1], dtype=torch.bool, device=dev), tm], 1)
        else:
            tm = torch.ones(tk.shape[:2], dtype=torch.bool, device=dev)
        kw = dict(tokens=tk, tok_mask=tm, state_n=STg[c], base_n=Bg[c])
        if text_dim:
            o = OBJg[r].clamp(min=0)
            w = WORDg[r].clamp(min=0) if wording is None else torch.full_like(o, wording)
            kw.update(sent=SENTg[o, w], has_sent=(k != 2).float(), instr=INSTRg[LIDg[c]])
        return kw, tgt

    tr = np.flatnonzero(row_train)
    tr0 = tr[kind[tr] == 0]
    tr0nz = tr0[np.abs(T[chunk[tr0]]).max(axis=(1, 2)) > 0]
    tr1, tr2 = tr[kind[tr] == 1], tr[kind[tr] == 2]
    nb = [a.bs // 2, a.bs // 4, a.bs - a.bs // 2 - a.bs // 4]

    def sample_rows():
        k = int(round(nb[0] * a.nz_frac))
        return [np.concatenate([rng.choice(tr0nz, k), rng.choice(tr0, nb[0] - k)]), rng.choice(tr1, nb[1]),
                rng.choice(tr2, nb[2])]

    model = ResidualFlowTok(tok_dim=TOK.shape[-1], n_tok=TOK.shape[1], text_dim=text_dim, horizon=T.shape[1],
                            layers=a.layers).to(dev)
    ema = copy.deepcopy(model).eval()
    print(f"trainable params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + np.cos(np.pi * s / a.steps)))
    stats = dict(st_m=st_m, st_s=st_s, b_m=b_m, b_s=b_s, scale=scale)

    def save(name):
        torch.save(dict(model=ema.state_dict(), backbone=a.backbone,
                        cfg=dict(tok_dim=int(TOK.shape[-1]), n_tok=int(TOK.shape[1]), text_dim=text_dim,
                                 horizon=int(T.shape[1]), layers=a.layers),
                        stats={k: np.asarray(v).tolist() for k, v in stats.items()},
                        pca=os.path.abspath(f"{a.pg}/pca.npz"), text_bank=os.path.abspath(f"{a.siglip}/text_bank.npz"),
                        holdout=a.holdout), os.path.join(a.out, name))

    # ---- validation sets ----
    vrng = np.random.default_rng(1)
    val = {}
    if a.holdout:
        hc = held_chunk[chunk] & ok_row
        nzr = np.abs(T[chunk]).max(axis=(1, 2)) > 0
        new_layout = hc & (kind == 0) & (obj != hobj) & nzr
        new_obj = hc & (kind == 0) & (obj == hobj) & nzr
        for name, m in (("newlayout_seenobj", new_layout), ("unseen_obj_milk", new_obj),
                        ("absent", hc & (kind == 1)), ("null", hc & (kind == 2))):
            idx = np.flatnonzero(m)
            val[name] = vrng.choice(idx, min(1500, len(idx)), replace=False) if len(idx) else idx
    val["train_fit"] = vrng.choice(tr0nz, 1500, replace=False)

    def evaluate():
        ema.eval()
        res = {}
        for name, idx in val.items():
            if not len(idx):
                continue
            P_, G_ = [], []
            for i in range(0, len(idx), 250):
                kw, tgt = batch(idx[i:i + 250])
                P_.append(ema.sample(n=4, **kw).cpu().numpy()); G_.append(tgt.cpu().numpy())
            p = np.concatenate(P_).reshape(len(idx), -1); t = np.concatenate(G_).reshape(len(idx), -1)
            pn, tn = np.linalg.norm(p, axis=1), np.linalg.norm(t, axis=1)
            if tn.mean() > 0:
                r = dict(R2=1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum(),
                         cos=((p * t).sum(1) / (pn * tn + 1e-9)).mean(), ratio=pn.mean() / tn.mean(),
                         mse=((p - t) ** 2).sum(1).mean())
            else:
                r = dict(mag=pn.mean())
            res[name] = {k: round(float(v), 4) for k, v in r.items()}
        return res

    log = open(os.path.join(a.out, "train_log.jsonl"), "w")
    t0, run = time.time(), None
    for step in range(1, a.steps + 1):
        model.train()
        loss = 0.0
        for rows in sample_rows():
            kw, tgt = batch(rows)
            loss = loss + model.loss(tgt, **kw) * len(rows) / a.bs
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
            if step % 5000 == 0:
                rec["val"] = evaluate()
                save(f"residual_{step // 1000}k.pt")
            print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    save("residual.pt")
    res = evaluate()
    print("final", res, flush=True)
    json.dump(res, open(os.path.join(a.out, "fit.json"), "w"))


if __name__ == "__main__":
    main()
