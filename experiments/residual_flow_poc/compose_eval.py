"""Exp 3 follow-ups (inference only, existing `main` model):
  stack_AB   base -> +ΔA(A clause) -> base' -> +ΔB(B clause; conditioned on base')
  stack_BA   the reverse order
  sum        ΔA_sample + ΔB_sample, both conditioned on the original base (near-trivial floor)
Writes runs/eval_compose/rollouts.jsonl (same record format as evaluate.py).
    python compose_eval.py --runs runs --out runs/eval_compose
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import FlowNet, sample  # noqa: E402
from src.texts import COMBINED  # noqa: E402
from src.traj import K  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--out", default="runs/eval_compose")
ap.add_argument("--workers", type=int, default=5)
ap.add_argument("--n_samples", type=int, default=2)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

TE = np.load("data/test.npz", allow_pickle=True)
Z = np.load("data/text_emb.npz", allow_pickle=True)
E = dict(zip(Z["texts"], Z["emb"]))
scenes, base, feats = TE["scenes"], TE["base"].astype(np.float64), TE["feats"]
NS = len(base)
st = dict(np.load(os.path.join(args.runs, "main", "stats.npz")))
model = FlowNet(use_text=True).to(dev)
model.load_state_dict(torch.load(os.path.join(args.runs, "main", "ema.pt"), map_location=dev))
model.eval()
T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
s_n = T((feats - st["s_mu"]) / st["s_sd"])


def gen(base_traj, text, seed):
    b = T((base_traj.reshape(NS, -1) - st["b_mu"]) / st["b_sd"])
    txt = T(np.repeat(E[text][None], NS, 0))
    g = torch.Generator(device=dev).manual_seed(seed)
    x = sample(model, s_n, b, [(txt, torch.zeros(NS, dtype=torch.bool, device=dev))], generator=g).cpu().numpy()
    return (x * st["r_sd"] + st["r_mu"]).reshape(NS, K, 6).astype(np.float64)


jobs = []
seed = 10_000
for si in range(args.n_samples):
    for j, (t, (ca, cb)) in enumerate(COMBINED):
        dA = gen(base, ca, seed); seed += 1
        dB = gen(base, cb, seed); seed += 1
        dB2 = gen(base + dA, cb, seed); seed += 1      # B conditioned on the A-corrected base
        dA2 = gen(base + dB, ca, seed); seed += 1      # A conditioned on the B-corrected base
        for cond, res in ((f"stack_AB|combined{j}", dA + dB2), (f"stack_BA|combined{j}", dB + dA2),
                          (f"sum|combined{j}", dA + dB)):
            for i in range(NS):
                jobs.append((dict(cond=cond, exp="exp3b", scene=i, sample=si, model="main", text=t), res[i]))
# references so analyze.py can compute deviation from base
for i in range(NS):
    jobs.append((dict(cond="base", exp="ref", scene=i, sample=0, model=None, text=None), np.zeros((K, 6))))
print(f"{len(jobs)} rollouts", flush=True)


def run(job):
    from src.sim import PandaScene
    meta, res = job
    i = meta["scene"]
    r = PandaScene(scenes[i]).rollout(base[i] + res, record=True)
    tr = r.pop("traj")
    meta = dict(meta); meta.update(r)
    meta["exec_grip"] = np.round(tr[:, 4:7], 4).tolist()
    meta["exec_tilt"] = np.round(tr[:, 3], 2).tolist()
    meta["corr_pos_mean"] = float(np.linalg.norm(res[:, :3], axis=-1).mean())
    meta["corr_pos_max"] = float(np.linalg.norm(res[:, :3], axis=-1).max())
    meta["corr_rot_mean_deg"] = float(np.degrees(np.linalg.norm(res[:, 3:], axis=-1).mean()))
    return meta


t0 = time.time()
with Pool(args.workers) as pool, open(os.path.join(args.out, "rollouts.jsonl"), "w") as f:
    for n, rec in enumerate(pool.imap(run, jobs, chunksize=4)):
        f.write(json.dumps(rec) + "\n")
print("done", time.time() - t0)
