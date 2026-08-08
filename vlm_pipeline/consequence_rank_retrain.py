"""Rank-loss retrain of the LS consequence model's progress head (#43 E1
follow-up, 2026-08-07).

Motivation: the E1 compose arm (critic ranks all K repair-certified
candidates by predicted progress; job 43160056) scored 67.3/3.3 vs the
production GT arm's 70.7/4.0 -- large opposing per-task swings (t6 5->9 but
t2 7->2, t7 9->5). The progress head was the ONE failed G1 gate (held-out
Spearman 0.5974 vs 0.60 bar) and compose leans entirely on it. The selector
consumes an ORDERING of candidates, not a regression value, so we retrain
with an added same-episode pairwise logistic ranking loss on the progress
head (all other heads/losses unchanged) and re-run the identical G1
evaluation on the SAME held-out split as `consequence_train/ls_model`.

Gate (pre-registered here): held-out Spearman must clear 0.60 AND improve on
the 0.5974 reference by >= 0.05 to justify any GPU rerun; err_ratio/AUC must
not regress past their bars (0.5 / 0.90).

Usage:
  python consequence_rank_retrain.py \
      --data_dir ../consequence_train/ls_data \
      --ref_model_dir ../consequence_train/ls_model \
      --out_dir ../consequence_train/ls_model_rank \
      [--rank_weight 1.0] [--rank_margin 0.005]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import List, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import consequence_model as cm  # noqa: E402


def rank_loss_for_batch(progress_pred, batch, idx_np, samples, rank_margin: float):
    """Pairwise logistic ranking loss over same-episode sample pairs in the
    batch whose progress targets are both valid and differ by more than
    `rank_margin` (meters of distance-reduction). Returns (loss, n_pairs)."""
    import torch

    ep_keys = [samples[i].episode_key for i in idx_np]
    mask = batch["progress_mask"].numpy() > 0
    target = batch["progress_target"].numpy()
    by_ep = {}
    for bi, key in enumerate(ep_keys):
        if mask[bi]:
            by_ep.setdefault(key, []).append(bi)
    left, right, sign = [], [], []
    for _, idxs in by_ep.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                d = target[i] - target[j]
                if abs(d) > rank_margin:
                    left.append(i)
                    right.append(j)
                    sign.append(1.0 if d > 0 else 0.0)
    if not left:
        return None, 0
    li = torch.tensor(left, dtype=torch.long)
    ri = torch.tensor(right, dtype=torch.long)
    lab = torch.tensor(sign, dtype=torch.float32)
    diff = progress_pred[li] - progress_pred[ri]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(diff, lab)
    return loss, len(left)


def train_member_ranked(
    train_samples: Sequence, held_samples: Sequence, norm, seed: int,
    rank_weight: float, rank_margin: float, max_epochs: int, patience: int,
    lr: float, batch_size: int,
):
    """Mirrors cm.train_ensemble's single-member loop, adding the rank loss.
    Batches are episode-contiguous (shuffle episodes, concatenate their
    sample indices) so same-episode pairs actually co-occur in a batch.
    Early stopping on held-out TOTAL loss (base + rank), best-state restore."""
    import torch

    torch.manual_seed(seed)
    model = cm.ConsequenceModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_batch = cm.samples_to_tensors(train_samples, norm)
    held_batch = cm.samples_to_tensors(held_samples, norm) if held_samples else None

    ep_to_idx = {}
    for i, s in enumerate(train_samples):
        ep_to_idx.setdefault(s.episode_key, []).append(i)
    ep_keys_list = list(ep_to_idx.keys())

    rng = np.random.default_rng(seed)
    best_held = float("inf")
    best_state = None
    since = 0
    history = []
    for epoch in range(max_epochs):
        order = []
        for k in rng.permutation(len(ep_keys_list)):
            order.extend(ep_to_idx[ep_keys_list[k]])
        order = np.asarray(order)
        model.train()
        ep_loss, nb = 0.0, 0
        for bstart in range(0, len(order), batch_size):
            idx = order[bstart:bstart + batch_size]
            batch = {k: v[idx] for k, v in train_batch.items()}
            opt.zero_grad()
            base_loss, _ = cm.compute_loss(model, batch)
            progress = model(batch["eef"], batch["grip"], batch["entity_rel"],
                             batch["entity_mask"], batch["domain_onehot"],
                             batch["action_window"])[3]
            rl, npairs = rank_loss_for_batch(progress, batch, idx, train_samples, rank_margin)
            loss = base_loss + (rank_weight * rl if rl is not None else 0.0)
            loss.backward()
            opt.step()
            ep_loss += float(loss)
            nb += 1
        ep_loss /= max(nb, 1)

        model.eval()
        with torch.no_grad():
            if held_batch is not None:
                h_base, _ = cm.compute_loss(model, held_batch)
                h_prog = model(held_batch["eef"], held_batch["grip"],
                               held_batch["entity_rel"], held_batch["entity_mask"],
                               held_batch["domain_onehot"], held_batch["action_window"])[3]
                h_rl, _ = rank_loss_for_batch(
                    h_prog, held_batch, np.arange(len(held_samples)),
                    held_samples, rank_margin)
                held_loss = float(h_base) + (rank_weight * float(h_rl) if h_rl is not None else 0.0)
            else:
                held_loss = ep_loss
        history.append((ep_loss, held_loss))
        if held_loss < best_held - 1e-6:
            best_held, best_state, since = held_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, dict(seed=seed, epochs_run=len(history), best_held_loss=best_held)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ref_model_dir", required=True,
                    help="existing model dir whose split.json is reused (comparability)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rank_weight", type=float, default=1.0)
    ap.add_argument("--rank_margin", type=float, default=0.005)
    ap.add_argument("--n_members", type=int, default=3)
    ap.add_argument("--max_epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    import torch  # noqa: F401 - fail early if unavailable

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = pathlib.Path(args.ref_model_dir)

    episodes = cm.load_episodes(pathlib.Path(args.data_dir))
    train_keys, held_keys = cm.load_split(ref_dir / "split.json")
    train_eps = [e for e in episodes if e.key in set(train_keys)]
    held_eps = [e for e in episodes if e.key in set(held_keys)]
    train_samples = cm.build_samples(train_eps)
    held_samples = cm.build_samples(held_eps)
    print(f"train {len(train_samples)} / held {len(held_samples)} samples "
          f"({len(train_eps)}/{len(held_eps)} episodes, split from {ref_dir})")
    norm = cm.compute_norm_stats(train_samples)

    models: List = []
    histories = []
    for m in range(args.n_members):
        model, hist = train_member_ranked(
            train_samples, held_samples, norm, seed=m,
            rank_weight=args.rank_weight, rank_margin=args.rank_margin,
            max_epochs=args.max_epochs, patience=args.patience,
            lr=args.lr, batch_size=args.batch_size)
        import torch
        torch.save({"state_dict": model.state_dict(), "seed": m},
                   out_dir / f"consequence_model_member{m}.pt")
        models.append(model)
        histories.append(hist)
        print(f"member {m}: {hist}")

    (out_dir / "norm_stats.json").write_text(json.dumps(norm.to_json(), indent=2))
    cm.save_split(out_dir / "split.json", train_keys, held_keys)
    (out_dir / "train_history.json").write_text(json.dumps(histories, indent=2))

    report = cm.evaluate_g1(held_samples, models, norm)
    report["rank_weight"] = args.rank_weight
    report["rank_margin"] = args.rank_margin
    (out_dir / "g1_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
