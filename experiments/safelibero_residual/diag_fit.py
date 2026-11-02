"""Training-set fit of residual checkpoints: R2, |pred| vs |target|, cosine, per row kind.
    python diag_fit.py data/residual_c0 runs/res_text_c0/residual.pt runs/res_notext_c0/residual.pt
"""
import sys, numpy as np, torch
sys.path.insert(0, ".")
dev = "cuda"
import texts as TX
from policy import ResidualFlow
from train_residual import load
F,B,R,T = load(sys.argv[1])
bank = TX.TextBank()
rng = np.random.default_rng(1)
def model(p):
    ck = torch.load(p, map_location="cpu"); m = ResidualFlow(feat_dim=ck["feat_dim"], text_dim=768 if ck["variant"]=="text" else 0)
    m.load_state_dict(ck["model"]); return m.eval().to(dev), ck["scale"]
pres = np.flatnonzero((R[:,2]==0) & (np.abs(T).max(axis=(1,2))>0))
pz = np.flatnonzero((R[:,2]==0) & (np.abs(T).max(axis=(1,2))==0))
absn = np.flatnonzero(R[:,2]==1)
for name in sys.argv[2:]:
    m, sc = model(name)
    for lab, ids in [("present_nz", pres), ("present_zero", pz), ("absent", absn)]:
        ids = rng.choice(ids, 1500, replace=False)
        rows = R[ids]; c = rows[:,0]
        text = np.stack([bank(TX.train_texts(TX.OBJ[o])[0]) for o in rows[:,1]]) if m.text_dim else np.zeros((len(ids),0),np.float32)
        f = torch.from_numpy(F[c].astype(np.float32)).to(dev); b = torch.from_numpy(B[c].astype(np.float32)).to(dev); t = torch.from_numpy(text.astype(np.float32)).to(dev)
        preds = torch.stack([m.sample(f,b,t) for _ in range(4)]).mean(0).cpu().numpy()/sc
        tg = T[ids].astype(np.float32)
        err = ((preds-tg)**2).sum(); var = ((tg-tg.mean(0))**2).sum()
        pn = np.linalg.norm(preds.reshape(len(ids),-1),axis=1); tn = np.linalg.norm(tg.reshape(len(ids),-1),axis=1)
        cos = (preds.reshape(len(ids),-1)*tg.reshape(len(ids),-1)).sum(1)/(pn*tn+1e-9)
        print(name.split('/')[1], lab, f"R2 {1-err/max(var,1e-9):.3f}  |pred| {pn.mean():.3f}  |tgt| {tn.mean():.3f}  cos {cos[tn>0].mean() if (tn>0).any() else float('nan'):.3f}", flush=True)
