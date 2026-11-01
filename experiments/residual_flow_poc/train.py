"""Train the text-conditioned residual flow (and its ablations).

Every variant sees the same (scene, target) stream:
  40% A-target with a random TRAIN_A phrasing
  40% B-target with a random TRAIN_B phrasing
  20% zero target with the empty instruction (null token)
Variants differ only in what text the network gets:
  main      the phrasing above
  notext    nothing (null token always)
  shuffled  a phrasing drawn independently of the target class
            (A/B/null in the same 40/40/20 proportions)
No combined A+B instruction or target is ever used.
    python train.py --variant main --out runs/main
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import FlowNet  # noqa: E402
from src.texts import TRAIN_A, TRAIN_B, TRAIN_AB, NULL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["main", "notext", "shuffled", "main_ab"], required=True)
ap.add_argument("--data_dir", default="data")
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--bs", type=int, default=1024)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--aug", action="store_true",
                help="post-hoc Exp-2 variant: within-class embedding mixup + gaussian noise (fixed a priori: lambda~U(0,1), sigma=0.02)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

D = np.load(os.path.join(args.data_dir, "train.npz"), allow_pickle=True)
Z = np.load(os.path.join(args.data_dir, "text_emb.npz"), allow_pickle=True)
E = dict(zip(Z["texts"], Z["emb"]))
N = len(D["base"])
base = D["base"].reshape(N, -1); dA = D["dA"].reshape(N, -1); dB = D["dB"].reshape(N, -1)
feats = D["feats"]
AB = args.variant == "main_ab"          # the only variant that ever reads the combined target
# class mix: 0=A 1=B 2=null 3=AB
MIX = [0.3, 0.3, 0.2, 0.2] if AB else [0.4, 0.4, 0.2, 0.0]
dAB = D["dAB"].reshape(N, -1) if AB else None

# normalization stats (train split only); residual stats over the class mixture
res_all = np.concatenate([dA, dB, np.zeros_like(dA[: N // 2])] + ([dAB] if AB else []))
stats = dict(
    b_mu=base.mean(0), b_sd=np.maximum(base.std(0), 1e-2),
    s_mu=feats.mean(0), s_sd=np.maximum(feats.std(0), 1e-2),
    r_mu=res_all.mean(0), r_sd=np.maximum(res_all.std(0), 1e-2),
)
np.savez(os.path.join(args.out, "stats.npz"), **stats)

T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
base_n = T((base - stats["b_mu"]) / stats["b_sd"])
feat_n = T((feats - stats["s_mu"]) / stats["s_sd"])
dA_n = T((dA - stats["r_mu"]) / stats["r_sd"])
dB_n = T((dB - stats["r_mu"]) / stats["r_sd"])
dAB_n = T((dAB - stats["r_mu"]) / stats["r_sd"]) if AB else None
zero_n = T((0 - stats["r_mu"]) / stats["r_sd"])
embA = T(np.stack([E[t] for t in TRAIN_A])); embB = T(np.stack([E[t] for t in TRAIN_B]))
embAB = T(np.stack([E[t] for t in TRAIN_AB])) if AB else None

model = FlowNet(use_text=args.variant != "notext").to(dev)
ema = FlowNet(use_text=args.variant != "notext").to(dev)
ema.load_state_dict(model.state_dict())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)


def draw_cls(n):
    u = torch.rand(n, device=dev)
    c = torch.cumsum(torch.tensor(MIX, device=dev), 0)
    return torch.searchsorted(c, u, right=True).clamp(max=3)


def draw_text(cls, n):
    """cls: (n,) int 0=A 1=B 2=null 3=AB -> (txt (n,768), is_null (n,))"""
    txt = torch.zeros(n, 768, device=dev)
    ia = torch.randint(0, len(TRAIN_A), (n,), device=dev)
    ib = torch.randint(0, len(TRAIN_B), (n,), device=dev)
    ea, eb = embA[ia], embB[ib]
    if args.aug:
        lam = torch.rand(n, 1, device=dev)
        ea = lam * ea + (1 - lam) * embA[torch.randint(0, len(TRAIN_A), (n,), device=dev)]
        eb = lam * eb + (1 - lam) * embB[torch.randint(0, len(TRAIN_B), (n,), device=dev)]
        ea = torch.nn.functional.normalize(ea + 0.02 * torch.randn_like(ea), dim=-1)
        eb = torch.nn.functional.normalize(eb + 0.02 * torch.randn_like(eb), dim=-1)
    txt = torch.where((cls == 0)[:, None], ea, txt)
    txt = torch.where((cls == 1)[:, None], eb, txt)
    if AB:
        txt = torch.where((cls == 3)[:, None], embAB[torch.randint(0, len(TRAIN_AB), (n,), device=dev)], txt)
    return txt, cls == 2


log = []
t0 = time.time()
for step in range(args.steps):
    idx = torch.randint(0, N, (args.bs,), device=dev)
    cls = draw_cls(args.bs)
    x1 = torch.where((cls == 0)[:, None], dA_n[idx], torch.where((cls == 1)[:, None], dB_n[idx], zero_n.expand(args.bs, -1)))
    if AB:
        x1 = torch.where((cls == 3)[:, None], dAB_n[idx], x1)
    if args.variant == "shuffled":
        tcls = draw_cls(args.bs)
    else:
        tcls = cls
    txt, is_null = draw_text(tcls, args.bs)
    x0 = torch.randn_like(x1)
    t = torch.rand(args.bs, device=dev)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    v = model(xt, t, feat_n[idx], base_n[idx], txt, is_null)
    loss = ((v - (x1 - x0)) ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - 0.999)
    if step % 500 == 0 or step == args.steps - 1:
        log.append(dict(step=step, loss=float(loss), t=time.time() - t0))
        print(log[-1], flush=True)

torch.save(ema.state_dict(), os.path.join(args.out, "ema.pt"))
json.dump(dict(args=vars(args), log=log), open(os.path.join(args.out, "train_log.json"), "w"), indent=1)
print("done", args.out)
