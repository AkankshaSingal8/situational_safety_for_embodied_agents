"""Consequence-model v2 retrain (#43 improvement round, user-approved
2026-08-08): three targeted fixes, each aimed at a measured failure.

1. LABELS — composite-distance discounted progress. The v1 label
   (one-chunk reduction of d(eef, target)) is (a) myopic: the t7-class cells
   need detours whose immediate progress is negative; (b) stage-blind: after
   the grasp d(eef, target) ~= 0 and the label carries no signal. New label:
       D_t = d(eef_t, target_t) + d(target_t, dest_t)
       progress_t = sum_{k=1..M} gamma^{(k-1)/H} * (D_{t+k-1} - D_{t+k})
   (M = LOOKAHEAD_CHUNKS*H env steps, gamma = per-chunk discount). The
   composite covers reach and transport continuously with no stage switch;
   dest comes from the bddl goal (target_maps/ls_obstacle_avoidance_dest.json,
   auto-derived + task-id-verified against the target map). Tasks/records
   missing a dest fall back to D = d(eef, target).

2. TRAINING — percep-noise injection. The v1 critic collapsed at the no-GT
   tier because it trained on clean GT entity positions but is fed detector+
   depth positions (2-4 cm typical, occasional gross outliers). Each training
   batch perturbs entity_rel with N(0, sigma^2) per axis plus outlier
   corruption (prob p, magnitude U(lo, hi), random direction) — matched to
   the measured percep error stats. Injection happens in normalized feature
   space (scaled by 1/rel_std) so the stored tensors stay clean.

3. LOSS — same-episode pairwise logistic rank loss on the new labels
   (the selector consumes an ordering; ranking is the objective that fixed
   v1: Spearman 0.597 -> 0.754).

Pre-registered gates (held-out split identical to ls_model):
   clean-input progress Spearman >= 0.70 AND noisy-input Spearman >= 0.60
   AND err_ratio <= 0.5 AND contact AUC >= 0.90. FAIL on any -> no GPU eval.

Resume ratchet: completed member checkpoints are skipped on relaunch
(login-node reaper survival, same mechanism as consequence_rank_retrain).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import consequence_model as cm  # noqa: E402
from consequence_rank_retrain import rank_loss_for_batch  # noqa: E402

H = cm.H


def _spearman(a, b):
    ar = np.argsort(np.argsort(a)).astype(np.float64)
    br = np.argsort(np.argsort(b)).astype(np.float64)
    if ar.std() == 0 or br.std() == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def relabel_progress(samples: Sequence, episodes, target_map: Dict[int, str],
                     dest_map: Dict[int, str], lookahead_chunks: int,
                     gamma: float) -> int:
    """In-place: replaces each sample.progress_target with the composite-distance
    discounted label. Sample j of an episode anchors record j (build_samples:
    anchor = recs[i-1] for i = 1..n-H). Returns #labeled."""
    ep_by_key = {e.key: e for e in episodes}
    by_ep: Dict[tuple, List] = {}
    for s in samples:
        by_ep.setdefault(s.episode_key, []).append(s)
    m_steps = lookahead_chunks * H
    n_labeled = 0
    for key, ss in by_ep.items():
        ep = ep_by_key[key]
        recs = ep.records
        tname = target_map.get(ep.task)
        dname = dest_map.get(ep.task)
        if tname is None:
            for s in ss:
                s.progress_mask = 0.0
            continue
        # Composite distance series over the whole episode; carry last-known
        # entity positions across records with missing entries.
        D = np.full(len(recs), np.nan)
        t_pos = d_pos = None
        for t, r in enumerate(recs):
            eef = np.asarray(r["eef"], dtype=np.float64)
            p = cm._entity_pos(r, tname)
            if p is not None:
                t_pos = p
            if dname is not None:
                q = cm._entity_pos(r, dname)
                if q is not None:
                    d_pos = q
            if t_pos is None:
                continue
            d = float(np.linalg.norm(eef - t_pos))
            if d_pos is not None:
                d += float(np.linalg.norm(t_pos - d_pos))
            D[t] = d
        # forward-fill isolated nans so diffs stay defined
        for t in range(1, len(D)):
            if np.isnan(D[t]):
                D[t] = D[t - 1]
        w = gamma ** (np.arange(m_steps, dtype=np.float64) / H)
        for j, s in enumerate(ss):
            a = j  # anchor record index
            if np.isnan(D[a]):
                s.progress_mask = 0.0
                continue
            hi = min(a + m_steps, len(D) - 1)
            if hi <= a:
                s.progress_mask = 0.0
                continue
            diffs = D[a:hi] - D[a + 1:hi + 1]  # per-step reductions
            s.progress_target = float(np.dot(w[:len(diffs)], diffs))
            s.progress_mask = 1.0
            n_labeled += 1
    return n_labeled


def make_noise_fn(norm, sigma: float, outlier_p: float,
                  outlier_lo: float, outlier_hi: float, seed: int):
    """Returns f(entity_rel_norm_tensor, entity_mask_tensor) -> perturbed
    tensor. Noise is calibrated in METERS then scaled into the normalized
    feature space by 1/rel_std."""
    import torch
    inv_std = torch.tensor(1.0 / np.asarray(norm.rel_std), dtype=torch.float32)  # (3,)
    g = torch.Generator().manual_seed(seed)

    def f(rel, mask):
        # rel: (B, E, 3) normalized; mask: (B, E)
        base = torch.randn(rel.shape, generator=g) * sigma
        out_flag = (torch.rand(rel.shape[:2], generator=g) < outlier_p).float()
        mag = outlier_lo + (outlier_hi - outlier_lo) * torch.rand(rel.shape[:2], generator=g)
        direc = torch.randn(rel.shape, generator=g)
        direc = direc / direc.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        noise_m = base + direc * (mag * out_flag).unsqueeze(-1)
        return rel + noise_m * inv_std * mask.unsqueeze(-1)
    return f


def train_member_v2(train_samples, held_samples, norm, seed, args,
                    partial_path=None):
    import torch
    torch.manual_seed(seed)
    model = cm.ConsequenceModel()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_batch = cm.samples_to_tensors(train_samples, norm)
    held_batch = cm.samples_to_tensors(held_samples, norm)
    noise_fn = make_noise_fn(norm, args.noise_sigma, args.outlier_p,
                             args.outlier_lo, args.outlier_hi, seed=seed + 100)

    ep_to_idx: Dict[tuple, List[int]] = {}
    for i, s in enumerate(train_samples):
        ep_to_idx.setdefault(s.episode_key, []).append(i)
    ep_keys = list(ep_to_idx)
    rng = np.random.default_rng(seed)
    best_held, best_state, since = float("inf"), None, 0
    start_epoch = 0
    if partial_path is not None and partial_path.exists():
        # Epoch-level ratchet (login-node reaper kills runs mid-member):
        # resume model + optimizer + early-stop bookkeeping mid-training.
        ck = torch.load(partial_path, map_location="cpu")
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt_state"])
        best_held, since, start_epoch = ck["best_held"], ck["since"], ck["epoch"] + 1
        best_state = ck.get("best_state")
        rng = np.random.default_rng(seed + 1000 + start_epoch)
        print(f"  resumed member seed={seed} at epoch {start_epoch} "
              f"(best_held {best_held:.4f})")
    for epoch in range(start_epoch, args.max_epochs):
        order = []
        for k in rng.permutation(len(ep_keys)):
            order.extend(ep_to_idx[ep_keys[k]])
        order = np.asarray(order)
        model.train()
        for bstart in range(0, len(order), args.batch_size):
            idx = order[bstart:bstart + args.batch_size]
            batch = {k: v[idx] for k, v in train_batch.items()}
            batch = dict(batch)
            batch["entity_rel"] = noise_fn(batch["entity_rel"], batch["entity_mask"])
            opt.zero_grad()
            base_loss, _ = cm.compute_loss(model, batch)
            progress = model(batch["eef"], batch["grip"], batch["entity_rel"],
                             batch["entity_mask"], batch["domain_onehot"],
                             batch["action_window"])[3]
            rl, _ = rank_loss_for_batch(progress, batch, idx, train_samples,
                                        args.rank_margin)
            loss = base_loss + (args.rank_weight * rl if rl is not None else 0.0)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            h_base, _ = cm.compute_loss(model, held_batch)
            h_prog = model(held_batch["eef"], held_batch["grip"],
                           held_batch["entity_rel"], held_batch["entity_mask"],
                           held_batch["domain_onehot"], held_batch["action_window"])[3]
            h_rl, _ = rank_loss_for_batch(h_prog, held_batch,
                                          np.arange(len(held_samples)),
                                          held_samples, args.rank_margin)
            held_loss = float(h_base) + (args.rank_weight * float(h_rl)
                                          if h_rl is not None else 0.0)
        if held_loss < best_held - 1e-6:
            best_held = held_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= args.patience:
                break
        if partial_path is not None:
            torch.save({"state_dict": model.state_dict(),
                        "opt_state": opt.state_dict(),
                        "best_state": best_state, "best_held": best_held,
                        "since": since, "epoch": epoch}, partial_path)
    if best_state is not None:
        model.load_state_dict(best_state)
    if partial_path is not None and partial_path.exists():
        partial_path.unlink()
    return model, dict(seed=seed, best_held_loss=best_held)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ref_model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_map", required=True)
    ap.add_argument("--dest_map", required=True)
    ap.add_argument("--lookahead_chunks", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--rank_weight", type=float, default=1.0)
    ap.add_argument("--rank_margin", type=float, default=0.01)
    ap.add_argument("--noise_sigma", type=float, default=0.02)
    ap.add_argument("--outlier_p", type=float, default=0.10)
    ap.add_argument("--outlier_lo", type=float, default=0.05)
    ap.add_argument("--outlier_hi", type=float, default=0.15)
    ap.add_argument("--n_members", type=int, default=2)
    ap.add_argument("--max_epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    import torch

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmap = {int(k): v for k, v in json.loads(pathlib.Path(args.target_map).read_text()).items()}
    dmap = {int(k): v for k, v in json.loads(pathlib.Path(args.dest_map).read_text()).items()}

    episodes = cm.load_episodes(pathlib.Path(args.data_dir))
    train_keys, held_keys = cm.load_split(pathlib.Path(args.ref_model_dir) / "split.json")
    train_eps = [e for e in episodes if e.key in set(train_keys)]
    held_eps = [e for e in episodes if e.key in set(held_keys)]
    train_samples = cm.build_samples(train_eps, tmap)
    held_samples = cm.build_samples(held_eps, tmap)
    nl_t = relabel_progress(train_samples, train_eps, tmap, dmap,
                            args.lookahead_chunks, args.gamma)
    nl_h = relabel_progress(held_samples, held_eps, tmap, dmap,
                            args.lookahead_chunks, args.gamma)
    print(f"relabeled train {nl_t}/{len(train_samples)}  held {nl_h}/{len(held_samples)}")
    if nl_t == 0:
        raise SystemExit("no relabeled train samples -- check target/dest maps")
    norm = cm.compute_norm_stats(train_samples)

    models, histories = [], []
    for m in range(args.n_members):
        ckpt = out_dir / f"consequence_model_member{m}.pt"
        if ckpt.exists():
            model = cm.ConsequenceModel()
            model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
            models.append(model)
            histories.append(dict(seed=m, resumed=True))
            print(f"member {m}: resumed")
            continue
        model, hist = train_member_v2(train_samples, held_samples, norm, m, args,
                                      partial_path=out_dir / f"member{m}_partial.pt")
        torch.save({"state_dict": model.state_dict(), "seed": m}, ckpt)
        models.append(model)
        histories.append(hist)
        print(f"member {m}: {hist}")

    (out_dir / "norm_stats.json").write_text(json.dumps(norm.to_json(), indent=2))
    cm.save_split(out_dir / "split.json", train_keys, held_keys)
    (out_dir / "train_history.json").write_text(json.dumps(histories, indent=2))

    # ---- gates: standard G1 (clean) + noisy-input progress Spearman -------
    report = cm.evaluate_g1(held_samples, models, norm)
    held_batch = cm.samples_to_tensors(held_samples, norm)
    noise_fn = make_noise_fn(norm, args.noise_sigma, args.outlier_p,
                             args.outlier_lo, args.outlier_hi, seed=1234)
    noisy_rel = noise_fn(held_batch["entity_rel"], held_batch["entity_mask"])
    preds = []
    with torch.no_grad():
        for model in models:
            model.eval()
            preds.append(model(held_batch["eef"], held_batch["grip"], noisy_rel,
                               held_batch["entity_mask"], held_batch["domain_onehot"],
                               held_batch["action_window"])[3].numpy())
    mp = np.mean(preds, axis=0)
    pm = np.array([s.progress_mask for s in held_samples]) > 0
    pt = np.array([s.progress_target for s in held_samples])
    noisy_rho = _spearman(mp[pm], pt[pm])
    clean_rho = report["per_domain"]["LS"]["spearman"]
    gate = (clean_rho >= 0.70 and noisy_rho >= 0.60
            and report["per_domain"]["LS"]["err_ratio"] <= 0.5
            and report["per_domain"]["LS"]["auc"] >= 0.90)
    report["v2"] = dict(noisy_spearman=noisy_rho, clean_spearman=clean_rho,
                        gate_pass=bool(gate),
                        lookahead_chunks=args.lookahead_chunks, gamma=args.gamma,
                        noise_sigma=args.noise_sigma, outlier_p=args.outlier_p)
    (out_dir / "g1_report.json").write_text(json.dumps(report, indent=2))
    print(f"V2 GATE: clean_rho={clean_rho:.4f} noisy_rho={noisy_rho:.4f} "
          f"-> {'PASS' if gate else 'FAIL'}")


if __name__ == "__main__":
    main()
