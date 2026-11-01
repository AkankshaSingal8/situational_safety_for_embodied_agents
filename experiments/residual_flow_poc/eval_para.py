"""Post-hoc Exp 2 follow-up: score `main` and `main_aug` on training phrasings,
HELDOUT (contaminated for main_aug), fresh HELDOUT2, and (main_aug only) the Exp 3
direct combined sentences. 1 flow sample per scene per phrasing.
    python eval_para.py --runs runs --out runs/eval_para
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
from src.texts import TRAIN_A, TRAIN_B, HELDOUT_A, HELDOUT_B, HELDOUT2_A, HELDOUT2_B, COMBINED  # noqa: E402
from src.traj import K  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--out", default="runs/eval_para")
ap.add_argument("--workers", type=int, default=5)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

TE = np.load("data/test.npz", allow_pickle=True)
Z = np.load("data/text_emb.npz", allow_pickle=True)
E = dict(zip(Z["texts"], Z["emb"]))
scenes, base, feats = TE["scenes"], TE["base"].astype(np.float64), TE["feats"]
NS = len(base)
T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731


def load(v):
    st = dict(np.load(os.path.join(args.runs, v, "stats.npz")))
    m = FlowNet(use_text=True).to(dev)
    m.load_state_dict(torch.load(os.path.join(args.runs, v, "ema.pt"), map_location=dev))
    return m.eval(), st


def gen(v, text, seed):
    m, st = MODELS[v]
    s = T((feats - st["s_mu"]) / st["s_sd"])
    b = T((base.reshape(NS, -1) - st["b_mu"]) / st["b_sd"])
    g = torch.Generator(device=dev).manual_seed(seed)
    x = sample(m, s, b, [(T(np.repeat(E[text][None], NS, 0)), torch.zeros(NS, dtype=torch.bool, device=dev))], generator=g)
    return (x.cpu().numpy() * st["r_sd"] + st["r_mu"]).reshape(NS, K, 6).astype(np.float64)


MODELS = {v: load(v) for v in ("main", "main_aug")}
sets = [("train", "A", TRAIN_A), ("train", "B", TRAIN_B), ("heldout", "A", HELDOUT_A), ("heldout", "B", HELDOUT_B),
        ("heldout2", "A", HELDOUT2_A), ("heldout2", "B", HELDOUT2_B)]
jobs, seed = [], 20_000
for v in MODELS:
    for split, cls, L in sets:
        for j, t in enumerate(L):
            res = gen(v, t, seed); seed += 1
            for i in range(NS):
                jobs.append((dict(model=v, split=split, cls=cls, idx=j, text=t, scene=i), res[i]))
for j, (t, _) in enumerate(COMBINED):
    res = gen("main_aug", t, seed); seed += 1
    for i in range(NS):
        jobs.append((dict(model="main_aug", split="combined", cls="AB", idx=j, text=t, scene=i), res[i]))
print(len(jobs), "rollouts", flush=True)


def run(job):
    from src.sim import PandaScene
    meta, res = job
    r = PandaScene(scenes[meta["scene"]]).rollout(base[meta["scene"]] + res)
    meta = dict(meta); meta.update(r)
    meta["corr_pos_mean"] = float(np.linalg.norm(res[:, :3], axis=-1).mean())
    meta["corr_rot_mean_deg"] = float(np.degrees(np.linalg.norm(res[:, 3:], axis=-1).mean()))
    return meta


t0 = time.time()
with Pool(args.workers) as pool, open(os.path.join(args.out, "rollouts.jsonl"), "w") as f:
    for rec in pool.imap(run, jobs, chunksize=4):
        f.write(json.dumps(rec) + "\n")
print("done", time.time() - t0)
