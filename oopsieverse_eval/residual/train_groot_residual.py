"""Train the GR00T residual expert on shield data (port of experiments/safelibero_residual/train_expert.py, ex8 mix).

Data: data_r0/<task>/seed*.npz from run_groot_stock_damagesim.py --shield --save_data (per chunk: 3 cameras,
16-d state, base xyz (16x3), executed xyz (16x3), flag). Episodes are kept only on safe success (task reached
and every tracked link still at 100), as in ex2.
Rows (ex8_lora mix corr=0.30, safe=0.40, wrong=0.10, none=0.20):
  corr  : true sentence, a chunk the shield replaced      -> target = executed - base
  safe  : true sentence, a chunk the shield left alone     -> target 0
  wrong : another task's sentence, any chunk (half corr)   -> target 0
  none  : instruction only, any chunk (half corr)          -> target 0
Optimiser as ex8_lora: AdamW b2 0.95, wd 1e-10; lr head 2e-5, LoRA 1e-4, new layer 1e-3; 500 warmup; fixed steps.
Run inside the groot_srv_u24 env (Ubuntu 24.04 sandbox).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from groot_residual import CAMS, SENTENCES, GrootResidual, prompt  # noqa: E402

MIX = dict(corr=0.30, safe=0.40, wrong=0.10, none=0.20)


def excluded_seeds(records_glob):
    """(task, seed) of collection episodes where the shield was off (worker scene != live scene)."""
    out = set()
    for f in glob.glob(records_glob) if records_glob else []:
        for e in json.load(open(f)):
            if e.get("clone_mismatch") or e.get("error"):
                out.add((e["task_name"], int(e["seed"])))
    return out


def load_episodes(roots, val_seeds, exclude=frozenset(), episodes="safe", max_plain=12, seed=0):
    """episodes="safe": ex2 filter (only safe-success episodes, every chunk kept).
    episodes="all" (r1): every collected episode; from episodes that are not safe successes keep only the corrected
    chunks plus up to max_plain chunks the shield left alone (memory). A chunk counts as a correction only when the
    shield picked a candidate whose 8-step lookahead was damage-free (cand not min_* / hold_fail fallback)."""
    tr, va = [], []
    rng = np.random.default_rng(seed)
    for root in roots.split(","):
        for f in sorted(glob.glob(os.path.join(root, "*", "seed*.npz"))):
            z = np.load(f, allow_pickle=True)
            hf = z["health_final"]
            safe_ep = bool(z["success_any"]) and (hf.size == 0 or float(hf.min()) >= 100 - 1e-6)
            if episodes == "safe" and not safe_ep:
                continue
            task = os.path.basename(os.path.dirname(f))
            seed_ = int(os.path.basename(f)[4:-4])
            if (task, seed_) in exclude:
                continue
            flag = z["flag"].astype(bool)
            cand = np.array([str(c) for c in z["cand"]]) if "cand" in z.files else np.array(["?"] * len(flag))
            fallback = np.array([c.startswith("min_") or c == "hold_fail" for c in cand])
            res = z["exec_xyz"] - z["base_xyz"]
            corr = flag & ~fallback & (np.abs(res).max(axis=(1, 2)) > 1e-6)
            if safe_ep or episodes == "safe":
                keep = np.arange(len(flag))
            else:
                plain = np.where(~flag)[0]
                plain = rng.choice(plain, size=min(max_plain, len(plain)), replace=False) if len(plain) else plain
                keep = np.sort(np.concatenate([np.where(corr)[0], plain]).astype(int))
                if not len(keep):
                    continue
            ep = dict(task=task, seed=seed_, lang=str(z["lang"]), safe_ep=safe_ep,
                      cams={c: z[c][keep] for c in CAMS}, state=z["state"][keep], base=z["base_xyz"][keep],
                      exe=z["exec_xyz"][keep], flag=flag[keep])
            ep["res"] = ep["exe"] - ep["base"]
            ep["corr"] = np.where(corr[keep])[0]
            # fallback chunks are flagged but not corrections: never used as corr, and not as "safe" (target 0)
            ep["flag"] = ep["flag"] | fallback[keep]
            (va if seed_ in val_seeds else tr).append(ep)
    return tr, va


def _pick(rng, eps):
    """ex8 --balance_suites: task first (uniform), then an episode of that task."""
    tasks = sorted({e["task"] for e in eps})
    t = tasks[rng.integers(len(tasks))]
    pool = [e for e in eps if e["task"] == t]
    return pool[rng.integers(len(pool))]


def make_batch(eps, rng, bs):
    kinds = rng.choice(list(MIX), size=bs, p=list(MIX.values()))
    corr_eps = [e for e in eps if len(e["corr"])]
    ims, sts, txs, bas, tgs, ks = [], [], [], [], [], []
    for k in kinds:
        use_corr = k == "corr" or (k in ("wrong", "none") and rng.random() < 0.5)
        if use_corr and corr_eps:
            e = _pick(rng, corr_eps); c = int(e["corr"][rng.integers(len(e["corr"]))])
        else:
            e = _pick(rng, eps)
            safe_idx = np.where(~e["flag"])[0] if k == "safe" else np.arange(len(e["flag"]))
            c = int(safe_idx[rng.integers(len(safe_idx))]) if len(safe_idx) else int(rng.integers(len(e["flag"])))
        if k == "corr" and not e["flag"][c]:
            k = "safe"
        sent = {"corr": SENTENCES[e["task"]], "safe": SENTENCES[e["task"]], "none": "",
                "wrong": SENTENCES[rng.choice([t for t in SENTENCES if t != e["task"]])]}[k]
        tgt = e["res"][c] if k == "corr" else np.zeros_like(e["res"][c])
        ims.append({cam: e["cams"][cam][c] for cam in CAMS}); sts.append(e["state"][c])
        txs.append(prompt(e["lang"], sent)); bas.append(e["base"][c]); tgs.append(tgt); ks.append(k)
    return ims, sts, txs, np.stack(bas), np.stack(tgs), ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="comma-separated data roots")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--lr_new", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--val_seeds", default="1090-1099")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="")
    ap.add_argument("--episodes", default="safe", choices=["safe", "all"])
    ap.add_argument("--gate_w", type=float, default=0.0, help="weight of the correction-needed gate BCE (class-balanced)")
    ap.add_argument("--max_plain", type=int, default=12)
    ap.add_argument("--exclude_records", default="/data/asingal2/oopsieverse/residual/collect_r0/*/episodes.json",
                    help="glob of collection episodes.json; clone_mismatch / error episodes are skipped")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lo, hi = map(int, a.val_seeds.split("-"))
    ex = excluded_seeds(a.exclude_records)
    tr, va = load_episodes(a.data, set(range(lo, hi + 1)), ex, a.episodes, a.max_plain, a.seed)
    print(f"excluded (shield off / error): {len(ex)}", flush=True)
    n_corr = sum(len(e["corr"]) for e in tr)
    print(f"episodes train {len(tr)} val {len(va)}; corrected chunks {n_corr}; per task eps "
          f"{ {t: sum(e['task'] == t for e in tr) for t in SENTENCES} } corr "
          f"{ {t: int(sum(len(e['corr']) for e in tr if e['task'] == t)) for t in SENTENCES} }", flush=True)
    assert tr and n_corr, "no episodes with corrections"
    R = np.concatenate([e["res"][e["corr"]].reshape(-1, 3) for e in tr if len(e["corr"])])
    R = R[np.abs(R).max(1) > 1e-6]
    B = np.concatenate([e["base"].reshape(-1, 3) for e in tr])
    torch.manual_seed(a.seed)
    m = GrootResidual()
    if a.init:
        m.load(a.init)
    else:
        m.scale, m.b_scale = float(1.0 / (R.std() + 1e-6)), float(1.0 / (B.std() + 1e-6))
    print(f"scale {m.scale:.3f} b_scale {m.b_scale:.3f}", flush=True)
    groups = {"lora": [], "new": [], "head": []}
    for n, p in m.named_parameters():
        if p.requires_grad:
            groups["lora" if "lora" in n else "new" if n.startswith("cond_proj") else "head"].append(p)
    lrs = dict(lora=a.lr_lora, new=a.lr_new, head=a.lr)
    print({k: sum(p.numel() for p in v) / 1e6 for k, v in groups.items()}, "M trainable", flush=True)
    opt = torch.optim.AdamW([dict(params=v, lr=lrs[k]) for k, v in groups.items() if v],
                            betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
    rng, vrng = np.random.default_rng(a.seed), np.random.default_rng(12345)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    json.dump(dict(args=vars(a), scale=m.scale, b_scale=m.b_scale, n_train=len(tr), n_val=len(va), n_corr=n_corr),
              open(os.path.join(a.out, "config.json"), "w"), indent=1)
    m.train()
    t0 = time.time()
    for step in range(1, a.steps + 1):
        ims, sts, txs, bas, tgs, _ = make_batch(tr, rng, a.bs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m.loss(ims, sts, txs, bas, tgs, gate_w=a.gate_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for v in groups.values() for p in v], 1.0)
        opt.step(); sched.step()
        rec = dict(step=step, loss=float(loss), sec=round(time.time() - t0, 1))
        if step % 100 == 0 and va:
            m.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vl, gp = [], []
                for _ in range(4 if a.gate_w else 2):
                    vl.append(float(m.loss(*make_batch(va, vrng, a.bs)[:5], gate_w=a.gate_w)))
                    if a.gate_w:
                        gp += list(zip(*m._last_gate))
            rec["val_loss"] = float(np.mean(vl)); m.train()
            if gp:
                pos = [g for g, y in gp if y]; neg = [g for g, y in gp if not y]
                rec["gate_bal_acc"] = round(0.5 * float(np.mean([g >= 0.5 for g in pos] or [0])) +
                                            0.5 * float(np.mean([g < 0.5 for g in neg] or [0])), 3)
                rec["gate_p_pos"] = round(float(np.mean(pos)), 3) if pos else None
                rec["gate_p_neg"] = round(float(np.mean(neg)), 3) if neg else None
        if step % 50 == 0 or "val_loss" in rec:
            print(rec, flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step % a.save_every == 0 or step == a.steps:
            m.save(os.path.join(a.out, f"step_{step}.pt"), extra=dict(step=step))
    m.save(os.path.join(a.out, "last.pt"), extra=dict(step=a.steps))
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
