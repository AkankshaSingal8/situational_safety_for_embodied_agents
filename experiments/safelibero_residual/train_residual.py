"""Train the text-conditioned residual flow (or its no-text control) on collected oracle data.

    python train_residual.py --data data/residual_train --out runs/res_text  --variant text
    python train_residual.py --data data/residual_train --out runs/res_notext --variant notext
    python train_residual.py --data d1,d2 --out runs/res_img --variant text_img \
        --scenes data/scenes_train.npz --holdout_layouts 20-24 --base_ckpt runs/base/ema_last.pt
Rows are drawn 50% present-obstacle / 25% absent-obstacle / 25% null. A present or absent
row gets a random training wording naming its object; null gets the zero vector. The
no-text model sees the same rows without the sentence, so it must hedge between "detour"
and "do nothing" for the same state.

*_img variants add trainable image encoders (initialised from the base policy's trunks, lr x0.1)
and need agent_/wrist_ frames saved by collect_residual.py --save_images.
--holdout_layouts drops every chunk whose scene uses those layouts from training and reports
validation fit (R2, |pred|/|target|, cosine on present non-zero rows; magnitude on zero rows) on them.
"""
import argparse
import copy
import glob
import json
import os
import time

import numpy as np
import torch

import texts as TX
from policy import ResidualFlow, ResidualFlowImg


def load(data_dir, images=False):
    F, B, R, T, CS, IA, IW = [], [], [], [], [], [], []
    off = 0
    files = sum((sorted(glob.glob(os.path.join(d, "part_*.npz"))) for d in data_dir.split(",")), [])
    for f in files:
        z = np.load(f)
        rows = z["rows"].copy(); rows[:, 0] += off
        F.append(z["feats"]); B.append(z["base"]); R.append(rows); T.append(z["targets"])
        CS.append(z["chunk_scene"] if "chunk_scene" in z else np.full(len(z["feats"]), -1, np.int32))
        if images:
            wid = os.path.basename(f)[5:-4]
            IA.append(np.load(os.path.join(os.path.dirname(f), f"agent_{wid}.npy")))
            IW.append(np.load(os.path.join(os.path.dirname(f), f"wrist_{wid}.npy")))
            assert len(IA[-1]) == len(z["feats"]), f
        off += len(z["feats"])
    out = [np.concatenate(x) for x in (F, B, R, T, CS)]
    if images:
        out += [np.concatenate(IA), np.concatenate(IW)]
    return out


class Fit:
    """Validation metrics on a fixed row subset (present non-zero / present zero / absent)."""

    def __init__(self, R, T, rng, n=2000):
        nz = np.abs(T).max(axis=(1, 2)) > 0
        self.sets = {"present_nz": np.flatnonzero((R[:, 2] == 0) & nz),
                     "present_zero": np.flatnonzero((R[:, 2] == 0) & ~nz),
                     "absent": np.flatnonzero(R[:, 2] == 1)}
        self.sets = {k: (rng.choice(v, min(n, len(v)), replace=False) if len(v) else v) for k, v in self.sets.items()}

    @torch.no_grad()
    def __call__(self, predict, T, scale):
        out = {}
        for k, ids in self.sets.items():
            if not len(ids):
                continue
            p = np.concatenate([predict(ids[i:i + 256]) for i in range(0, len(ids), 256)]) / scale
            tg = T[ids].astype(np.float32)
            pf, tf = p.reshape(len(ids), -1), tg.reshape(len(ids), -1)
            pn, tn = np.linalg.norm(pf, axis=1), np.linalg.norm(tf, axis=1)
            if k == "present_nz":
                out["R2"] = float(1 - ((pf - tf) ** 2).sum() / ((tf - tf.mean(0)) ** 2).sum())
                out["ratio"] = float(pn.mean() / tn.mean())
                out["cos"] = float(((pf * tf).sum(1) / (pn * tn + 1e-9)).mean())
            else:
                out[f"mag_{k}"] = float(pn.mean())
        return {k: round(v, 3) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="one dir or comma-separated dirs (aggregated)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["text", "notext", "text_img", "notext_img"], default="text")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc_lr_mult", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenes", default="", help="scenes npz the data was collected on (for --holdout_layouts)")
    ap.add_argument("--holdout_layouts", default="", help="lo-hi layouts excluded from training, used for validation")
    ap.add_argument("--holdout_objects", default="", help="comma-separated obstacle names never seen in training")
    ap.add_argument("--eval_only", default="", help="skip training; report fit of this residual.pt")
    ap.add_argument("--bn_batch", action="store_true", help="with --eval_only: BatchNorm uses batch statistics")
    ap.add_argument("--bn_free", action="store_true", help="*_img: let encoder BatchNorm stats update (old behaviour)")
    ap.add_argument("--base_ckpt", default="runs/base/ema_last.pt", help="encoder init for *_img")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda"
    img = a.variant.endswith("_img")
    use_text = a.variant.startswith("text")
    data = load(a.data, images=img)
    F, B, R, T, CS = data[:5]
    # train / validation split by layout
    val_chunk = np.zeros(len(F), bool)
    if a.holdout_layouts:
        from eval_sl import load_scenes
        meta = load_scenes(a.scenes)[1]
        lo, hi = map(int, a.holdout_layouts.split("-"))
        assert (CS >= 0).all(), "holdout needs chunk_scene (data collected after 2026-11-01)"
        lay = np.array([meta[s]["layout"] for s in CS])
        val_chunk = (lay >= lo) & (lay <= hi)
    ho_obj = [o for o in a.holdout_objects.split(",") if o]
    if ho_obj:
        # unseen-object split: no chunk from a scene that contains a held-out obstacle, and no row whose
        # sentence names one (absent-object rows included), is used for training
        from eval_sl import load_scenes
        meta = load_scenes(a.scenes)[1]
        ho_idx = [TX.OBJ.index(o) for o in ho_obj]
        has_ho = np.array([any(o in ho_obj for o in meta[s]["obstacles"]) for s in CS])
        val_chunk |= has_ho
        print(f"held-out objects {ho_obj}: {int(has_ho.sum())} chunks from scenes containing them", flush=True)
    tr_rows = ~val_chunk[R[:, 0]]
    if ho_obj:
        named_ho = np.isin(R[:, 1], ho_idx)
        tr_rows &= ~named_ho
    va_rows = val_chunk[R[:, 0]]
    Rtr, Rva = R[tr_rows], R[va_rows]
    bank = TX.TextBank()
    temb = np.stack([np.stack([bank(s) for s in TX.train_texts(o)]) for o in TX.OBJ])  # (6, W, 768)
    nz = T[R[:, 2] == 0].astype(np.float32)
    act_mask = np.abs(nz).max(axis=(1, 2)) > 0
    scale = float(1.0 / (nz[act_mask].std() + 1e-6)) if act_mask.any() else 1.0
    print(f"chunks {len(F)} ({int(val_chunk.sum())} held out), rows {len(R)} (train {len(Rtr)}), present rows "
          f"{int((R[:, 2] == 0).sum())} ({act_mask.mean():.3f} of them non-zero), target scale {scale:.2f}", flush=True)
    Fg = torch.from_numpy(F.astype(np.float32)).to(dev)
    Bg = torch.from_numpy(B.astype(np.float32)).to(dev)
    Tg = torch.from_numpy(T.astype(np.float32) * scale).to(dev)
    Eg = torch.from_numpy(temb).to(dev)
    if img:
        IAg, IWg = torch.from_numpy(data[5]).to(dev), torch.from_numpy(data[6]).to(dev)
    by_kind = [np.flatnonzero(Rtr[:, 2] == k) for k in range(3)]
    Rg = torch.from_numpy(Rtr.astype(np.int64)).to(dev)
    tdim = 768 if use_text else 0
    if img:
        model = ResidualFlowImg(feat_dim=F.shape[1], text_dim=tdim).to(dev)
        model.init_from_base(torch.load(a.base_ckpt, map_location="cpu")["model"])
        enc = set(id(p) for p in model.encoder_params())
        groups = [{"params": [p for p in model.parameters() if id(p) not in enc], "lr": a.lr},
                  {"params": model.encoder_params(), "lr": a.lr * a.enc_lr_mult}]
    else:
        model = ResidualFlow(feat_dim=F.shape[1], text_dim=tdim).to(dev)
        groups = [{"params": list(model.parameters()), "lr": a.lr}]
    if a.eval_only:
        model.load_state_dict(torch.load(a.eval_only, map_location="cpu")["model"])
    ema = copy.deepcopy(model).eval()
    if img and not a.bn_free:
        model.freeze_bn()  # encoder BN keeps the base policy's statistics: train == eval behaviour
    if a.eval_only and a.bn_batch:
        for mod in ema.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.train(); mod.momentum = 0.0
    opt = torch.optim.AdamW(groups, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + np.cos(np.pi * s / a.steps)))
    scaler = torch.cuda.amp.GradScaler(enabled=img)
    W = temb.shape[1]

    def text_for(rows, random_wording=True):
        obj = rows[:, 1].clamp(min=0)
        wsel = torch.randint(0, W, (len(rows),), device=dev) if random_wording else torch.zeros_like(obj)
        return Eg[obj, wsel] * (rows[:, 2] != 2).float().unsqueeze(1)

    def predictor(m, Rsub):
        Rs = torch.from_numpy(Rsub.astype(np.int64)).to(dev)

        def predict(ids):
            rows = Rs[torch.from_numpy(ids).to(dev)]
            c = rows[:, 0]
            t = text_for(rows, random_wording=False)
            outs = []
            for _ in range(4):
                if img:
                    outs.append(m.sample_img(IAg[c], IWg[c], Fg[c], Bg[c], t))
                else:
                    outs.append(m.sample(Fg[c], Bg[c], t))
            return torch.stack(outs).mean(0).float().cpu().numpy()
        return predict

    tr_idx = np.flatnonzero(tr_rows)  # Rtr[j] == R[tr_idx[j]]
    t0, run = time.time(), None
    log = open(os.path.join(a.out, "train_log.jsonl"), "w")
    for step in range(1, 0 if a.eval_only else a.steps + 1):
        nb = [a.bs // 2, a.bs // 4, a.bs - a.bs // 2 - a.bs // 4]
        idx = np.concatenate([rng.choice(by_kind[k], nb[k]) for k in range(3) if len(by_kind[k])])
        rows = Rg[torch.from_numpy(idx).to(dev)]
        c = rows[:, 0]
        text = text_for(rows)
        tgt = Tg[torch.from_numpy(tr_idx[idx]).to(dev)]
        with torch.autocast("cuda", dtype=torch.float16, enabled=img):
            if img:
                loss = model.loss_img(IAg[c], IWg[c], Fg[c], Bg[c], text, tgt)
            else:
                loss = model.loss(Fg[c], Bg[c], text, tgt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.999).add_(pm.detach(), alpha=0.001)
            for be, bm in zip(ema.buffers(), model.buffers()):  # BatchNorm running stats (image encoders)
                be.copy_(bm)
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 1000 == 0:
            rec = dict(step=step, loss=round(run, 5), min=round((time.time() - t0) / 60, 1))
            print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    if not a.eval_only:
      torch.save(dict(model=ema.state_dict(), variant=a.variant, scale=scale, feat_dim=F.shape[1]),
               os.path.join(a.out, "residual.pt"))
      print("saved", flush=True)
    # fit on training rows and on held-out-layout rows
    res = {}
    for name, Rsub, Tsub in [("train", Rtr, T[tr_rows]), ("val", Rva, T[va_rows])]:
        if len(Rsub):
            res[name] = Fit(Rsub, Tsub, np.random.default_rng(1))(predictor(ema, Rsub), Tsub, scale)
            print(f"fit {name}: {res[name]}", flush=True)
    json.dump(res, open(os.path.join(a.out, "fit_eval.json" if a.eval_only else "fit.json"), "w"))


if __name__ == "__main__":
    main()
