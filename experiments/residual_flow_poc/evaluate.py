"""Generate corrected trajectories for every condition and execute them in MuJoCo.

Writes runs/eval/rollouts.jsonl (one line per rollout, metrics from sim state)
and runs/eval/waypoints.npz (commanded trajectories, for deviation analysis).
    python evaluate.py --runs runs --out runs/eval --workers 5
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
from src.texts import TRAIN_A, TRAIN_B, HELDOUT_A, HELDOUT_B, COMBINED, NULL  # noqa: E402
from src.traj import K  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--out", default="runs/eval")
ap.add_argument("--workers", type=int, default=5)
ap.add_argument("--n_scenes", type=int, default=100)
ap.add_argument("--s_exp1", type=int, default=3)
ap.add_argument("--s_exp2", type=int, default=1)
ap.add_argument("--s_exp3", type=int, default=2)
ap.add_argument("--limit", type=int, default=0, help="debug: cap number of rollouts")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

TE = np.load("data/test.npz", allow_pickle=True)
TR = np.load("data/train.npz", allow_pickle=True)
Z = np.load("data/text_emb.npz", allow_pickle=True)
E = dict(zip(Z["texts"], Z["emb"]))
NS = min(args.n_scenes, len(TE["base"]))
scenes = TE["scenes"][:NS]
base = TE["base"][:NS]
feats = TE["feats"][:NS]


def load(variant):
    st = dict(np.load(os.path.join(args.runs, variant, "stats.npz")))
    m = FlowNet(use_text=variant != "notext").to(dev)
    m.load_state_dict(torch.load(os.path.join(args.runs, variant, "ema.pt"), map_location=dev))
    m.eval()
    return m, st


MODELS = {v: load(v) for v in ("main", "notext", "shuffled")}


def gen(variant, texts, seed, clauses=None):
    """texts: list (len NS) of instruction strings. clauses: list (len NS) of
    clause lists -> inference-time composition. Returns (NS, K, 6) residuals."""
    m, st = MODELS[variant]
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    s = T((feats - st["s_mu"]) / st["s_sd"])
    b = T((base.reshape(NS, -1) - st["b_mu"]) / st["b_sd"])
    g = torch.Generator(device=dev).manual_seed(seed)
    if clauses is None:
        txt = T(np.stack([E[t] for t in texts]))
        nul = torch.tensor([t == NULL for t in texts], device=dev)
        conds = [(txt, nul)]
    else:
        nc = len(clauses[0])
        conds = []
        for j in range(nc):
            cj = [c[j] for c in clauses]
            conds.append((T(np.stack([E[t] for t in cj])), torch.zeros(NS, dtype=torch.bool, device=dev)))
    x = sample(m, s, b, conds, generator=g).cpu().numpy()
    return (x * st["r_sd"] + st["r_mu"]).reshape(NS, K, 6)


jobs = []  # (meta dict, residual (K,6))


def add(cond, exp, residuals, sample_idx, texts=None, model=None):
    for i in range(NS):
        jobs.append((dict(cond=cond, exp=exp, scene=i, sample=sample_idx, model=model,
                          text=None if texts is None else texts[i]), residuals[i]))


rot = lambda L: [L[i % len(L)] for i in range(NS)]  # noqa: E731  phrasing rotated across scenes
zeros = np.zeros((NS, K, 6), np.float32)

# ---------------- references ----------------
add("base", "ref", zeros, 0)
add("oracle_A", "ref", TE["dA"][:NS], 0)
add("oracle_B", "ref", TE["dB"][:NS], 0)
add("oracle_AB", "ref", TE["dAB"][:NS], 0)
mA, mB = TR["dA"].mean(0), TR["dB"].mean(0)
add("mean_A", "exp1", np.repeat(mA[None], NS, 0), 0)
add("mean_B", "exp1", np.repeat(mB[None], NS, 0), 0)
add("mean_A+mean_B", "exp3", np.repeat((mA + mB)[None], NS, 0), 0)

# ---------------- Exp 1: text-conditioned correction ----------------
seed = 0
for si in range(args.s_exp1):
    for cond, texts in [("A", rot(TRAIN_A)), ("B", rot(TRAIN_B)), ("null", [NULL] * NS)]:
        add(f"main|{cond}", "exp1", gen("main", texts, seed), si, texts, "main"); seed += 1
    for cond, texts in [("A", rot(TRAIN_A)), ("B", rot(TRAIN_B))]:
        add(f"shuffled|{cond}", "exp1", gen("shuffled", texts, seed), si, texts, "shuffled"); seed += 1
    add("notext", "exp1", gen("notext", [NULL] * NS, seed), si, None, "notext"); seed += 1

# ---------------- Exp 2: unseen paraphrases ----------------
for si in range(args.s_exp2):
    for j, t in enumerate(HELDOUT_A):
        add(f"main|heldout_A{j}", "exp2", gen("main", [t] * NS, seed), si, [t] * NS, "main"); seed += 1
    for j, t in enumerate(HELDOUT_B):
        add(f"main|heldout_B{j}", "exp2", gen("main", [t] * NS, seed), si, [t] * NS, "main"); seed += 1

# ---------------- Exp 3: unseen combination ----------------
for si in range(args.s_exp3):
    for j, (t, cl) in enumerate(COMBINED):
        add(f"main|combined{j}", "exp3", gen("main", [t] * NS, seed), si, [t] * NS, "main"); seed += 1
        add(f"compose|combined{j}", "exp3", gen("main", None, seed, clauses=[cl] * NS), si, [t] * NS, "main"); seed += 1
        add(f"shuffled|combined{j}", "exp3", gen("shuffled", [t] * NS, seed), si, [t] * NS, "shuffled"); seed += 1

if args.limit:
    jobs = jobs[: args.limit]
print(f"{len(jobs)} rollouts queued", flush=True)
np.savez_compressed(os.path.join(args.out, "waypoints.npz"),
                    residual=np.stack([r for _, r in jobs]).astype(np.float32),
                    meta=np.array([json.dumps(m) for m, _ in jobs]))


def run(job):
    from src.sim import PandaScene
    meta, res = job
    i = meta["scene"]
    wp = base[i].astype(np.float64) + res.astype(np.float64)
    r = PandaScene(scenes[i]).rollout(wp, record=True)
    tr = r.pop("traj")
    meta = dict(meta)
    meta.update(r)
    meta["exec_grip"] = np.round(tr[:, 4:7], 4).tolist()   # executed grip-site path
    meta["exec_tilt"] = np.round(tr[:, 3], 2).tolist()     # executed mug tilt per waypoint
    # commanded correction magnitude
    meta["corr_pos_mean"] = float(np.linalg.norm(res[:, :3], axis=-1).mean())
    meta["corr_pos_max"] = float(np.linalg.norm(res[:, :3], axis=-1).max())
    meta["corr_rot_mean_deg"] = float(np.degrees(np.linalg.norm(res[:, 3:], axis=-1).mean()))
    return meta


t0 = time.time()
with Pool(args.workers) as pool, open(os.path.join(args.out, "rollouts.jsonl"), "w") as f:
    for n, rec in enumerate(pool.imap(run, jobs, chunksize=4)):
        f.write(json.dumps(rec) + "\n")
        if n % 500 == 0:
            print(f"{n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
print("done", time.time() - t0)
