"""Offline check of a residual checkpoint on held-out shield data: on corrected chunks, does the predicted xyz
residual point where the shield's correction points (cosine, magnitude ratio)? On uncorrected chunks, is it ~0?
usage: diag_residual.py --ckpt last.pt --data data_r0 [--val_seeds 1090-1099]"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from groot_residual import SENTENCES, GrootResidual, prompt, CAMS
from train_groot_residual import load_episodes

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--val_seeds", default="1090-1099"); ap.add_argument("--n", type=int, default=6)
a = ap.parse_args()
lo, hi = map(int, a.val_seeds.split("-")); vs = set(range(lo, hi + 1))
_, va = load_episodes(a.data, vs)
m = GrootResidual(); m.load(a.ckpt); m.eval()
rows = {"corr": [], "safe": []}
for e in va:
    nc = [c for c in range(len(e["flag"])) if c not in set(e["corr"])]
    for kind, idx in (("corr", list(e["corr"])), ("safe", nc[:: max(1, len(nc) // 6)])):
        for c in idx:
            im = {cam: e["cams"][cam][c] for cam in CAMS}
            preds = []
            for mode in ("true", "none"):
                tx = prompt(e["lang"], SENTENCES[e["task"]] if mode == "true" else "")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = np.mean([m.sample([im], [e["state"][c]], [tx], e["base"][c][None])[0] for _ in range(a.n)], 0)
                preds.append(p)
            t = e["res"][c]
            cos = float((preds[0] * t).sum() / (np.linalg.norm(preds[0]) * np.linalg.norm(t) + 1e-9))
            rows[kind].append((e["task"], np.linalg.norm(t), np.linalg.norm(preds[0]), np.linalg.norm(preds[1]), cos))
for k, r in rows.items():
    if not r:
        continue
    r = np.array([x[1:] for x in r])
    print(f"{k}: n={len(r)} |target| {r[:,0].mean():.4f} |pred true-text| {r[:,1].mean():.4f} "
          f"|pred no-text| {r[:,2].mean():.4f} cos(pred,target) {r[:,3].mean():.3f}")
for x in rows["corr"][:20]:
    print("  ", x[0], " ".join(f"{v:.4f}" for v in x[1:]))
