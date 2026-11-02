"""Train the base image flow policy on packed LIBERO-Spatial demos.

    python train_base.py --data DATA_DIR --out runs/base --steps 60000
Whole dataset sits on the GPU as uint8 (about 6 GB at 128 px), so there is no loader bottleneck.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from policy import BasePolicy

H = 16


def build_chunk_index(ep):
    """For each frame i, indices of the next H frames, clamped to the end of i's episode."""
    N = len(ep)
    end = np.zeros(N, np.int64)
    # last index of each episode
    change = np.flatnonzero(np.diff(ep))
    ends = np.append(change, N - 1)
    starts = np.insert(change + 1, 0, 0)
    for s, e in zip(starts, ends):
        end[s:e + 1] = e
    idx = np.arange(N)[:, None] + np.arange(H)[None]
    return np.minimum(idx, end[:, None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--save_every", type=int, default=20000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    L = lambda k: np.load(os.path.join(a.data, k + ".npy"), mmap_mode="r")  # noqa: E731
    agent = torch.from_numpy(np.ascontiguousarray(L("agent"))).to(dev)
    wrist = torch.from_numpy(np.ascontiguousarray(L("wrist"))).to(dev)
    prop = torch.from_numpy(np.ascontiguousarray(L("prop"))).to(dev)
    act = torch.from_numpy(np.ascontiguousarray(L("act"))).to(dev)
    task = torch.from_numpy(np.ascontiguousarray(L("task")).astype(np.int64)).to(dev)
    cidx = torch.from_numpy(build_chunk_index(np.asarray(L("ep")))).to(dev)
    N = len(act)
    amin, amax = torch.tensor(meta["act_min"], device=dev), torch.tensor(meta["act_max"], device=dev)
    pm, ps = torch.tensor(meta["prop_mean"], device=dev), torch.tensor(meta["prop_std"], device=dev)
    norm_a = lambda x: 2 * (x - amin) / (amax - amin + 1e-6) - 1  # noqa: E731
    print(f"frames {N}, episodes {meta['n_episodes']}, tasks {len(meta['langs'])}", flush=True)

    model = BasePolicy(len(meta["langs"]), pretrained=not a.no_pretrained).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
    scaler = torch.cuda.amp.GradScaler()
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    t0, run = time.time(), 0.0
    for step in range(1, a.steps + 1):
        i = torch.randint(0, N, (a.bs,), device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model.loss(agent[i], wrist[i], (prop[i] - pm) / ps, task[i], norm_a(act[cidx[i]]))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, pm_ in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm_.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 500 == 0:
            rec = dict(step=step, loss=round(run, 5), lr=sched.get_last_lr()[0], min=round((time.time() - t0) / 60, 1))
            print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if step % a.save_every == 0 or step == a.steps:
            torch.save(dict(model=ema.state_dict(), meta=meta, step=step, H=H), os.path.join(a.out, f"ema_{step}.pt"))
            torch.save(dict(model=ema.state_dict(), meta=meta, step=step, H=H), os.path.join(a.out, "ema_last.pt"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
