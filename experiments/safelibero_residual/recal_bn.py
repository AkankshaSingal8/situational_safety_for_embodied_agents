"""Recompute BatchNorm running stats of an image residual whose EMA kept stale BN buffers
(train_residual.py before 2026-11-01 evening copied parameters but not buffers into the EMA).
Forward-only, cumulative average over training chunks; reports train/val fit before and after.
    python recal_bn.py --ckpt runs/res_text_img_i1/residual.pt --data d1,d2 --scenes data/scenes_train.npz --holdout_layouts 20-24
Writes residual_bn.pt and fit_bn.json next to the checkpoint.
"""
import argparse, json, os
import numpy as np
import torch
import texts as TX
from policy import ResidualFlowImg
from train_residual import load, Fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--scenes", required=True); ap.add_argument("--holdout_layouts", default="20-24")
    ap.add_argument("--batches", type=int, default=300); ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    dev = "cuda"
    F, B, R, T, CS, IA, IW = load(a.data, images=True)
    from eval_sl import load_scenes
    meta = load_scenes(a.scenes)[1]
    lo, hi = map(int, a.holdout_layouts.split("-"))
    lay = np.array([meta[s]["layout"] for s in CS]); val_chunk = (lay >= lo) & (lay <= hi)
    tr_rows = ~val_chunk[R[:, 0]]
    ck = torch.load(a.ckpt, map_location="cpu")
    m = ResidualFlowImg(feat_dim=ck["feat_dim"], text_dim=768 if ck["variant"].startswith("text") else 0)
    m.load_state_dict(ck["model"]); m.to(dev).eval()
    scale = ck["scale"]
    Fg, Bg = torch.from_numpy(F.astype(np.float32)).to(dev), torch.from_numpy(B.astype(np.float32)).to(dev)
    IAg, IWg = torch.from_numpy(IA).to(dev), torch.from_numpy(IW).to(dev)
    bank = TX.TextBank()
    Eg = torch.from_numpy(np.stack([np.stack([bank(s) for s in TX.train_texts(o)]) for o in TX.OBJ])).to(dev)

    def fits():
        out = {}
        for name, mask in [("train", tr_rows), ("val", ~tr_rows)]:
            Rs_np, Ts = R[mask], T[mask]
            Rs = torch.from_numpy(Rs_np.astype(np.int64)).to(dev)

            def predict(ids):
                rows = Rs[torch.from_numpy(ids).to(dev)]; c = rows[:, 0]
                t = Eg[rows[:, 1].clamp(min=0), 0] * (rows[:, 2] != 2).float().unsqueeze(1)
                return torch.stack([m.sample_img(IAg[c], IWg[c], Fg[c], Bg[c], t) for _ in range(4)]).mean(0).cpu().numpy()
            out[name] = Fit(Rs_np, Ts, np.random.default_rng(1))(predict, Ts, scale)
        return out

    before = fits(); print("before", before, flush=True)
    bns = [mod for mod in m.modules() if isinstance(mod, torch.nn.BatchNorm2d)]
    for bn in bns:
        bn.reset_running_stats(); bn.momentum = None; bn.train()
    tr_chunks = np.flatnonzero(~val_chunk); rng = np.random.default_rng(0)
    with torch.no_grad():
        for _ in range(a.batches):
            c = torch.from_numpy(rng.choice(tr_chunks, a.bs)).to(dev)
            m._map(m.enc_a, IAg[c], False); m._map(m.enc_w, IWg[c], False)
    m.eval()
    after = fits(); print("after", after, flush=True)
    d = os.path.dirname(a.ckpt)
    torch.save(dict(ck, model=m.state_dict()), os.path.join(d, "residual_bn.pt"))
    json.dump(dict(before=before, after=after), open(os.path.join(d, "fit_bn.json"), "w"))


if __name__ == "__main__":
    main()
