"""v2 evaluation: every experiment on one split (train or test scenes), all in one place.

    python evaluate_v2.py --split test  --data_dir data_v2 --runs runs_v2 --out runs_v2/eval_test
    python evaluate_v2.py --split train --data_dir data_v2 --runs runs_v2 --out runs_v2/eval_train

split=train scores the first n_scenes of the training scenes (seen in training);
split=test scores the held-out test scenes. Conditions are identical across splits.

Conditions (cond names used by analyze_v2.py):
  refs   base, oracle_A, oracle_B, oracle_AB, mean_A, mean_B  (means over the training set)
  exp1   {main,main_ab}|{A,B,null}, shuffled|{A,B}, notext       (training phrasings)
  exp2   main|heldout_{A,B}{j}, main|heldout2_{A,B}{j}             (unseen paraphrases)
  exp3, no combined training (model `main`):
         main|combined{j}      the unseen sentence fed directly
         compose|combined{j}   v = v_null + sum_i (v_clause_i - v_null)
         stack_AB|combined{j}  base -> +dA(A clause) -> +dB(B clause, conditioned on the new base)
         stack_BA|combined{j}  the reverse order
         sum|combined{j}       dA + dB, both conditioned on the original base
         shuffled|combined{j}  no-information baseline
  exp3, with combined training (model `main_ab`):
         main_ab|combined{j}   the same six unseen sentences
         main_ab|trainAB{j}    the combined phrasings main_ab was trained on
  exp3, second unseen set COMBINED_FAR (worded far from TRAIN_AB):
         {main,stack_AB,stack_BA,sum,main_ab}|far{j}
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
from src.texts import (TRAIN_A, TRAIN_B, TRAIN_AB, HELDOUT_A, HELDOUT_B, HELDOUT2_A, HELDOUT2_B,  # noqa: E402
                       COMBINED, COMBINED_FAR, NULL)
from src.traj import K  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["train", "test"], required=True)
ap.add_argument("--data_dir", default="data_v2")
ap.add_argument("--runs", default="runs_v2")
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=5)
ap.add_argument("--n_scenes", type=int, default=100)
ap.add_argument("--s_exp1", type=int, default=2)
ap.add_argument("--s_exp3", type=int, default=2)
ap.add_argument("--limit", type=int, default=0, help="debug: cap number of rollouts")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

TR = np.load(os.path.join(args.data_dir, "train.npz"), allow_pickle=True)
SP = TR if args.split == "train" else np.load(os.path.join(args.data_dir, "test.npz"), allow_pickle=True)
Z = np.load(os.path.join(args.data_dir, "text_emb.npz"), allow_pickle=True)
E = dict(zip(Z["texts"], Z["emb"]))
NS = min(args.n_scenes, len(SP["base"]))
scenes = SP["scenes"][:NS]
base = SP["base"][:NS].astype(np.float64)
feats = SP["feats"][:NS]
TT = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731


def load(variant):
    st = dict(np.load(os.path.join(args.runs, variant, "stats.npz")))
    m = FlowNet(use_text=variant != "notext").to(dev)
    m.load_state_dict(torch.load(os.path.join(args.runs, variant, "ema.pt"), map_location=dev))
    return m.eval(), st


MODELS = {v: load(v) for v in ("main", "notext", "shuffled", "main_ab")}   # all four are required
print("models:", sorted(MODELS), "split:", args.split, "scenes:", NS, flush=True)


def gen(variant, texts, seed, base_traj=None, clauses=None):
    """texts: list (NS) of strings, or None with `clauses` (list of NS clause lists) for
    velocity composition. base_traj: (NS,K,6) base to condition on (default: the scene base).
    Returns (NS, K, 6) residuals in action units."""
    m, st = MODELS[variant]
    b0 = base if base_traj is None else base_traj
    s = TT((feats - st["s_mu"]) / st["s_sd"])
    b = TT((b0.reshape(NS, -1) - st["b_mu"]) / st["b_sd"])
    g = torch.Generator(device=dev).manual_seed(seed)
    if clauses is None:
        conds = [(TT(np.stack([E[t] for t in texts])), torch.tensor([t == NULL for t in texts], device=dev))]
    else:
        conds = [(TT(np.stack([E[c[j]] for c in clauses])), torch.zeros(NS, dtype=torch.bool, device=dev))
                 for j in range(len(clauses[0]))]
    x = sample(m, s, b, conds, generator=g).cpu().numpy()
    return (x * st["r_sd"] + st["r_mu"]).reshape(NS, K, 6).astype(np.float64)


jobs = []


def add(cond, exp, residuals, sample_idx, texts=None, model=None):
    for i in range(NS):
        jobs.append((dict(split=args.split, cond=cond, exp=exp, scene=i, sample=sample_idx, model=model,
                          text=None if texts is None else texts[i]), residuals[i]))


rot = lambda L: [L[i % len(L)] for i in range(NS)]  # noqa: E731  phrasing rotated across scenes
zeros = np.zeros((NS, K, 6))

# ---------------- references ----------------
add("base", "ref", zeros, 0)
add("oracle_A", "ref", SP["dA"][:NS], 0)
add("oracle_B", "ref", SP["dB"][:NS], 0)
add("oracle_AB", "ref", SP["dAB"][:NS], 0)
mA, mB = TR["dA"].mean(0), TR["dB"].mean(0)
add("mean_A", "exp1", np.repeat(mA[None], NS, 0), 0)
add("mean_B", "exp1", np.repeat(mB[None], NS, 0), 0)

# ---------------- Exp 1 ----------------
seed = 0
for si in range(args.s_exp1):
    for v in ("main", "main_ab"):
        for cond, texts in (("A", rot(TRAIN_A)), ("B", rot(TRAIN_B)), ("null", [NULL] * NS)):
            add(f"{v}|{cond}", "exp1", gen(v, texts, seed), si, texts, v); seed += 1
    for cond, texts in (("A", rot(TRAIN_A)), ("B", rot(TRAIN_B))):
        add(f"shuffled|{cond}", "exp1", gen("shuffled", texts, seed), si, texts, "shuffled"); seed += 1
    add("notext", "exp1", gen("notext", [NULL] * NS, seed), si, None, "notext"); seed += 1

# ---------------- Exp 2 ----------------
seed = 100_000
for tag, LA, LB in (("heldout", HELDOUT_A, HELDOUT_B), ("heldout2", HELDOUT2_A, HELDOUT2_B)):
    for cls, L in (("A", LA), ("B", LB)):
        for j, t in enumerate(L):
            add(f"main|{tag}_{cls}{j}", "exp2", gen("main", [t] * NS, seed), 0, [t] * NS, "main"); seed += 1

# ---------------- Exp 3 ----------------
seed = 200_000
for si in range(args.s_exp3):
    for j, (t, (ca, cb)) in enumerate(COMBINED):
        T = [t] * NS
        add(f"main|combined{j}", "exp3", gen("main", T, seed), si, T, "main"); seed += 1
        add(f"compose|combined{j}", "exp3", gen("main", None, seed, clauses=[[ca, cb]] * NS), si, T, "main"); seed += 1
        # common random numbers: each clause's second-pass sample reuses the noise of its
        # first-pass sample, so stack vs sum differs only by the re-conditioning
        sA, sB = seed, seed + 1; seed += 2
        dA = gen("main", [ca] * NS, sA)
        dB = gen("main", [cb] * NS, sB)
        dB2 = gen("main", [cb] * NS, sB, base_traj=base + dA)
        dA2 = gen("main", [ca] * NS, sA, base_traj=base + dB)
        add(f"stack_AB|combined{j}", "exp3", dA + dB2, si, T, "main")
        add(f"stack_BA|combined{j}", "exp3", dB + dA2, si, T, "main")
        add(f"sum|combined{j}", "exp3", dA + dB, si, T, "main")
        add(f"shuffled|combined{j}", "exp3", gen("shuffled", T, seed), si, T, "shuffled"); seed += 1
        add(f"main_ab|combined{j}", "exp3", gen("main_ab", T, seed), si, T, "main_ab"); seed += 1
# second unseen combined set, worded far from TRAIN_AB (same arms minus compose/shuffled)
seed = 300_000
for si in range(args.s_exp3):
    for j, (t, (ca, cb)) in enumerate(COMBINED_FAR):
        T = [t] * NS
        add(f"main|far{j}", "exp3", gen("main", T, seed), si, T, "main"); seed += 1
        sA, sB = seed, seed + 1; seed += 2
        dA = gen("main", [ca] * NS, sA)
        dB = gen("main", [cb] * NS, sB)
        dB2 = gen("main", [cb] * NS, sB, base_traj=base + dA)
        dA2 = gen("main", [ca] * NS, sA, base_traj=base + dB)
        add(f"stack_AB|far{j}", "exp3", dA + dB2, si, T, "main")
        add(f"stack_BA|far{j}", "exp3", dB + dA2, si, T, "main")
        add(f"sum|far{j}", "exp3", dA + dB, si, T, "main")
        add(f"main_ab|far{j}", "exp3", gen("main_ab", T, seed), si, T, "main_ab"); seed += 1
seed = 400_000
for j, t in enumerate(TRAIN_AB):
    add(f"main_ab|trainAB{j}", "exp3", gen("main_ab", [t] * NS, seed), 0, [t] * NS, "main_ab"); seed += 1

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
    r = PandaScene(scenes[i]).rollout(base[i] + res.astype(np.float32).astype(np.float64), record=True)
    tr = r.pop("traj")
    meta = dict(meta)
    meta.update(r)
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
        if n % 2000 == 0:
            print(f"{n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
print("done", time.time() - t0)
