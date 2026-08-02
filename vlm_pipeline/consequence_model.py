"""Consequence model: predicts H-step realized-eef-path, per-entity CONTACT,
HOLDING, and task-PROGRESS from a state + action-chunk, trained on logged
`transitions_*.jsonl` data (see `--log_transitions` in the two eval clients,
Task 1 of the 2026-08-01 consequence-steering-code plan).

Design doc: docs/superpowers/specs/2026-08-01-consequence-steering-design.md
Brief:      .superpowers/sdd/2026-08-01-consequence-steering-code/task-2-brief.md

Runs for real (training) in the `aegis` env. This module is also imported by
the CPU test suite (`openvla_libero_merged`), which may or may not have
torch installed; all torch-dependent names are guarded so a bare `import
consequence_model` never raises even when torch is absent — only calling a
torch-dependent function does.

Design decisions (documented per the brief):

- **Domain tag.** Two domains only: "SL" (SafeLIBERO, filename contains
  "safelibero") vs "LS" (LIBERO-Safety, everything else) — matches the two
  clients' filename schemes from Task 1
  (`transitions_{suite}_L{level}.jsonl` vs
  `transitions_{task_suite_name}_{safety_level}.jsonl`, and
  `task_suite_name` always starts with "safelibero").
- **Entity ordering / E=8 slots.** Per episode, the entity-slot ordering is
  the sorted (deterministic) union of entity names seen across all records
  in that episode, truncated to the first 8 names alphabetically if more
  than 8 appear (rare; documented, not expected to bind given typical
  scene sizes). Fewer than 8 -> remaining slots are zero-padded with
  mask=0. A slot with mask=0 at a given sample step contributes 0 to the
  relative-position input and is excluded from the CONTACT loss/AUC for
  that sample.
  **Task 3 (inference-time) interface contract:** this ordering is
  PER-EPISODE and the trunk is a flattened MLP over slots — it is NOT
  permutation-invariant across entities. Any caller that wants meaningful
  per-slot CONTACT logits at inference must reproduce the exact same
  `sorted(union of names seen)[:8]` construction (see `EpisodeRecords.
  entity_order`) for whatever "episode so far" context it is scoring, and
  must track which slot index corresponds to which entity name itself
  (the model has no notion of entity identity beyond slot position).
- **Sample anchor / windowing.** Per the Task 1 report's POST-step
  convention, `record[t].eef` is the state *after* `record[t].action`. A
  sample is anchored at record index `i` (i>=1): state s_i =
  (record[i-1].eef, record[i-1].grip, record[i-1].entities at s_i). The
  H-step action window is `record[i].action .. record[i+H-1].action`
  (translation components only, first 3 dims); the realized path target is
  `record[i+k].eef - s_i.eef` for k=0..H-1 (relative displacement from the
  anchor, matching the single-integrator baseline's cumsum-from-start
  convention). Requires `i-1 >= 0` and `i+H-1 <= last_index`; episodes
  shorter than H at the tail simply produce no more windows there (DROP,
  not mask) — simplest option consistent with the brief's "drop or mask"
  choice, and avoids polluting the path/contact loss with partially-fake
  padded actions.
- **CONTACT_i.** Tracked per-step over the window using each step's own
  entities dict (entities are logged alongside eef at the same record, so
  `record[i+k].entities[name]` is the entity position at the same instant
  as `record[i+k].eef`): CONTACT_i = 1 iff
  `min_k ||record[i+k].eef - record[i+k].entities[name_i]|| < 0.05` over
  the k where the entity is present that step; if the entity is absent
  from every step in the window, CONTACT_i for that sample/slot is masked
  out (not a hard negative — entity simply unobserved).
- **HOLDING.** `grip = record[i+H-1].grip = [q0, q1]`; width = q0 + q1
  (two-finger gripper qpos convention); label = 1 iff
  0.002 < width < 0.045.
- **PROGRESS.** `dist_to_target(s) = ||eef(s) - target_entity_pos(s)||`
  evaluated at the anchor (s_i, i.e. `record[i-1]`) and at the window end
  (`record[i+H-1]`); label = dist_to_target(s_i) - dist_to_target(end). If
  the episode's task has no `--target_map` entry, OR the target entity is
  absent at either endpoint, the progress label is masked for that sample
  (contact/holding labels are kept regardless, per the brief).
- **Normalization.** Positions/relative-positions and actions are
  z-scored using train-split statistics only (mean/std per feature
  dimension, computed once, saved to `norm_stats.json`, applied
  identically at eval/inference). Entity-relative positions (`rel_mean`/
  `rel_std`) and the H-step realized path (`path_std`, zero-mean/scale-only)
  are normalized with SEPARATE stats — they span different physical scales
  (entity distances ~0.1-0.5 m typical vs. per-window eef displacement
  ~0.01-0.2 m typical) and pooling them would silently de-weight the path
  regression loss against the entity-distance scale in the summed
  multi-head loss. Un-normalizing multiplies back by `path_std` so G1's
  terminal-point-error metric is reported in meters.
- **G1 per-domain gating (fix round 1, reviewer-mandated, binding).** The LS
  and SL data-collection servers run at DIFFERENT single-integrator
  calibrations (see `BASELINE_CONFIG`), so G1 metric (a) must not compare a
  domain's samples against the other domain's baseline. `evaluate_g1`
  computes all three G1 metrics SEPARATELY per domain present in the
  held-out set (`report["per_domain"][domain]`, each gated against the
  same fixed thresholds using that domain's own baseline calibration for
  metric (a)) plus a pooled cut (`report["pooled"]`, informational only —
  its metric (a) still uses each pooled sample's own domain calibration,
  so it is well-defined, just not what the gate decision is based on). The
  overall gate verdict (`report["pass"]`) is True iff EVERY domain present
  passes its own thresholds; one `G1[<domain>]: ...` line is printed per
  domain plus a final overall `G1: ...` line.
- **G1 unmeasurable vs failed.** Each per-domain (and the pooled) metrics
  dict reports `n_near_entity_samples`, `n_contact_positives`,
  `n_slots_scored`, `n_progress_labeled` alongside each metric, and an
  `"unmeasurable"` flag: if a held-out split happens to contain zero
  near-entity samples, zero contact positives, or fewer than 2
  progress-labeled samples (for that domain, or overall), the corresponding
  metric is NaN and the printed verdict reads `FAIL(unmeasurable)` rather
  than a bare `FAIL` — a no-GO from an unmeasured metric (e.g. an
  unlucky/too-small held-out split, or a domain absent from held-out
  entirely) must not be conflated with a no-GO from a genuinely bad model,
  since G1 failing halts all downstream GPU spend (design doc §5).
- **GRU vs flattened-MLP.** GRU over the H action steps: chosen because H
  is fixed at 10 (so either works) but a recurrent trunk generalizes to
  variable H later (e.g. shorter replan windows) without an architecture
  change, and naturally supports the cumulative-increment parameterization
  used for the path head (state carried forward step by step, like the
  single-integrator baseline it is meant to improve on).
- **Path head parameterization.** At each GRU step the model predicts an
  incremental displacement; the path is the cumsum of increments. This
  biases the model toward the single-integrator prior (a constant-velocity
  guess is trivial to represent) while still allowing learned correction.
- **Model size.** GRU hidden=64, context MLP hidden=64 -> per-member
  parameter count is on the order of 3-4x10^4, well under the 1M budget
  (see `count_parameters` used in the smoke test).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in torch-less envs
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


H = 10               # action-chunk / path horizon (steps)
E = 8                # entity slots
CONTACT_DIST = 0.05  # meters
HOLD_LO, HOLD_HI = 0.002, 0.045
DOMAINS = ("LS", "SL")

# Calibrated single-integrator baseline: (translation_scale, cmd_clip) PER
# DOMAIN. Only the LS-domain data-collection server runs exec-parity-guided
# (2.0, 0.025) -- see the "Guidance-arm note" in
# run_guided_libero_safety_eval.py:587-591 (--exec_parity's promoted
# guidance-arm config). SafeLIBERO-domain servers run
# openpi_guided/scripts/serve_policy_guided.py's PLAIN DEFAULTS
# (translation_scale=0.05, cmd_clip=1.0) -- exactly
# contextual_predictive_filter.PredictiveFilterConfig's defaults
# (translation_scale=0.050, max_translation_command=1.0), which is why that
# module is the correct citation for the SL regime specifically, not the LS
# one. Using the LS constants uniformly against SL-domain samples would
# compare the model to the WRONG actuation model on that domain (reviewer
# finding, fix round 1).
BASELINE_CONFIG = {
    "LS": (2.0, 0.025),
    "SL": (0.05, 1.0),
}
BASELINE_TRANSLATION_SCALE, BASELINE_CMD_CLIP = BASELINE_CONFIG["LS"]  # back-compat default
NEAR_ENTITY_RADIUS = 0.25  # G1 restricts the terminal-error comparison to
                            # samples with any entity within this radius


def baseline_params_for_domain(domain: str) -> Tuple[float, float]:
    """(translation_scale, cmd_clip) for `domain`; falls back to the LS
    calibration for any domain not in BASELINE_CONFIG (defensive; DOMAINS
    is currently exactly {LS, SL})."""
    return BASELINE_CONFIG.get(domain, BASELINE_CONFIG["LS"])


def _require_torch():
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "torch is required for this operation but is not installed in "
            "this environment. Training/evaluation must run in an env with "
            "torch (e.g. `aegis`)."
        )


def infer_domain(path: Path) -> str:
    """Two-domain heuristic from filename (see module docstring)."""
    name = Path(path).name.lower()
    return "SL" if "safelibero" in name else "LS"


# --------------------------------------------------------------------------
# Dataset parsing
# --------------------------------------------------------------------------

@dataclass
class EpisodeRecords:
    source_file: str
    task: int
    ep: int
    domain: str
    records: List[dict]  # sorted by t, each the raw JSONL dict

    @property
    def key(self) -> Tuple[str, int, int]:
        return (self.source_file, self.task, self.ep)

    def entity_order(self) -> List[str]:
        names = set()
        for r in self.records:
            names.update(r.get("entities", {}).keys())
        return sorted(names)[:E]


def load_episodes(data_dir: Path) -> List[EpisodeRecords]:
    """Load all `transitions_*.jsonl` under `data_dir` (recursive), grouped
    by (source_file, task, ep) and sorted by t within each group.

    Dedupes duplicate (task, ep, t) records WITHIN a source file (first
    occurrence wins, in file order), logging a warning per source file that
    had duplicates. A requeued SLURM task appending to the same transitions
    file (job resumed after a preemption/timeout without truncating prior
    output) would otherwise silently double up records at the same t,
    corrupting the H-step windows `build_samples` constructs from
    consecutive `t` indices (opportunistic fix, consequence-steering-code
    plan, fix round 2)."""
    data_dir = Path(data_dir)
    groups: Dict[Tuple[str, int, int], List[dict]] = {}
    for jf in sorted(data_dir.rglob("transitions_*.jsonl")):
        seen_t: Dict[Tuple[int, int, int], set] = {}
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (str(jf), int(rec["task"]), int(rec["ep"]))
                t = int(rec["t"])
                t_seen = seen_t.setdefault(key, set())
                if t in t_seen:
                    logging.warning(
                        "load_episodes: duplicate record at %s task=%s ep=%s t=%s "
                        "in %s -- keeping first occurrence, dropping the rest "
                        "(likely a requeued job re-appending to the same file)",
                        key, rec["task"], rec["ep"], t, jf)
                    continue
                t_seen.add(t)
                groups.setdefault(key, []).append(rec)
    episodes = []
    for (sf, task, ep), recs in groups.items():
        recs = sorted(recs, key=lambda r: r["t"])
        episodes.append(EpisodeRecords(
            source_file=sf, task=task, ep=ep,
            domain=infer_domain(Path(sf)), records=recs,
        ))
    return episodes


# --------------------------------------------------------------------------
# Sample construction
# --------------------------------------------------------------------------

@dataclass
class Sample:
    episode_key: Tuple[str, int, int]
    domain: str
    task: int
    eef: np.ndarray               # (3,)
    grip: np.ndarray              # (2,)
    entity_rel: np.ndarray        # (E, 3) relative to eef, masked-zero
    entity_mask: np.ndarray       # (E,) 1/0
    entity_names: List[str]       # length E (padded with "")
    action_window: np.ndarray     # (H, 3) translation-only
    path_target: np.ndarray       # (H, 3) relative displacement from eef
    contact_target: np.ndarray    # (E,) 0/1
    contact_mask: np.ndarray      # (E,) 1/0 (entity observed at least once)
    holding_target: float
    progress_target: float
    progress_mask: float          # 1.0 if valid else 0.0


def _entity_pos(rec: dict, name: str) -> Optional[np.ndarray]:
    v = rec.get("entities", {}).get(name)
    return None if v is None else np.asarray(v, dtype=np.float64)


def build_samples(
    episodes: Sequence[EpisodeRecords],
    target_map: Optional[Dict[int, str]] = None,
) -> List[Sample]:
    target_map = target_map or {}
    samples: List[Sample] = []
    for ep in episodes:
        recs = ep.records
        n = len(recs)
        names = ep.entity_order()
        padded_names = names + [""] * (E - len(names))
        target_name = target_map.get(ep.task)
        for i in range(1, n):
            if i + H - 1 > n - 1:
                break  # tail shorter than H: drop (documented decision)
            anchor = recs[i - 1]
            eef = np.asarray(anchor["eef"], dtype=np.float64)
            grip = np.asarray(anchor["grip"], dtype=np.float64)

            entity_rel = np.zeros((E, 3), dtype=np.float64)
            entity_mask = np.zeros((E,), dtype=np.float64)
            for slot, nm in enumerate(padded_names):
                if not nm:
                    continue
                pos = _entity_pos(anchor, nm)
                if pos is not None:
                    entity_rel[slot] = pos - eef
                    entity_mask[slot] = 1.0

            window = recs[i:i + H]
            action_window = np.array(
                [w["action"][:3] for w in window], dtype=np.float64)
            path_target = np.array(
                [np.asarray(w["eef"], dtype=np.float64) - eef for w in window],
                dtype=np.float64)

            contact_target = np.zeros((E,), dtype=np.float64)
            contact_mask = np.zeros((E,), dtype=np.float64)
            for slot, nm in enumerate(padded_names):
                if not nm:
                    continue
                dists = []
                for w in window:
                    pos = _entity_pos(w, nm)
                    if pos is None:
                        continue
                    dists.append(np.linalg.norm(
                        np.asarray(w["eef"], dtype=np.float64) - pos))
                if dists:
                    contact_mask[slot] = 1.0
                    contact_target[slot] = 1.0 if min(dists) < CONTACT_DIST else 0.0

            end_rec = window[-1]
            end_grip = np.asarray(end_rec["grip"], dtype=np.float64)
            width = float(end_grip[0] + end_grip[1])
            holding_target = 1.0 if (HOLD_LO < width < HOLD_HI) else 0.0

            progress_target = 0.0
            progress_mask = 0.0
            if target_name is not None:
                p0 = _entity_pos(anchor, target_name)
                p1 = _entity_pos(end_rec, target_name)
                if p0 is not None and p1 is not None:
                    d0 = float(np.linalg.norm(eef - p0))
                    end_eef = np.asarray(end_rec["eef"], dtype=np.float64)
                    d1 = float(np.linalg.norm(end_eef - p1))
                    progress_target = d0 - d1
                    progress_mask = 1.0

            samples.append(Sample(
                episode_key=ep.key, domain=ep.domain, task=ep.task,
                eef=eef, grip=grip,
                entity_rel=entity_rel, entity_mask=entity_mask,
                entity_names=padded_names,
                action_window=action_window, path_target=path_target,
                contact_target=contact_target, contact_mask=contact_mask,
                holding_target=holding_target,
                progress_target=progress_target, progress_mask=progress_mask,
            ))
    return samples


# --------------------------------------------------------------------------
# Episode-level train/held-out split
# --------------------------------------------------------------------------

def split_episodes(
    episodes: Sequence[EpisodeRecords],
    held_out_frac: float = 0.2,
    seed: int = 0,
) -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int]]]:
    keys = sorted(ep.key for ep in episodes)  # deterministic order pre-shuffle
    rng = random.Random(seed)
    shuffled = keys[:]
    rng.shuffle(shuffled)
    n_held = max(1, round(len(shuffled) * held_out_frac)) if shuffled else 0
    held = sorted(shuffled[:n_held])
    train = sorted(shuffled[n_held:])
    return train, held


def save_split(path: Path, train_keys, held_keys) -> None:
    payload = {
        "train": [list(k) for k in train_keys],
        "held_out": [list(k) for k in held_keys],
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_split(path: Path):
    payload = json.loads(Path(path).read_text())
    train = [tuple(k) for k in payload["train"]]
    held = [tuple(k) for k in payload["held_out"]]
    return train, held


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

@dataclass
class NormStats:
    """Train-split-only z-score stats. `rel_mean`/`rel_std` normalize
    entity-relative positions (`Sample.entity_rel`, meter-scale, ~0.1-0.5 m
    typical span). `path_std` is a SEPARATE scale-only (zero-mean, since
    path targets are already relative displacements centered near zero)
    normalizer for the H-step realized path (~0.01-0.2 m typical span) --
    kept distinct from `rel_std` so the path-regression loss isn't
    de-weighted by the larger entity-distance scale (both terms enter the
    summed multi-head loss, see `compute_loss`). `action_mean`/`action_std`
    normalize the H-step action-translation window fed to the GRU. Raw
    `eef`/`grip` are NOT normalized (kept in metric units in both the
    `Sample` objects and the tensors built by `samples_to_tensors` --
    `evaluate_g1`'s baseline/near-entity/terminal-error math is done in
    metric units directly off the `Sample` objects, not off normalized
    tensors)."""
    rel_mean: np.ndarray
    rel_std: np.ndarray
    path_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def to_json(self) -> dict:
        return {
            "rel_mean": self.rel_mean.tolist(), "rel_std": self.rel_std.tolist(),
            "path_std": self.path_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }

    @staticmethod
    def from_json(d: dict) -> "NormStats":
        return NormStats(
            rel_mean=np.asarray(d["rel_mean"]), rel_std=np.asarray(d["rel_std"]),
            path_std=np.asarray(d["path_std"]),
            action_mean=np.asarray(d["action_mean"]),
            action_std=np.asarray(d["action_std"]),
        )


def compute_norm_stats(samples: Sequence[Sample], eps: float = 1e-6) -> NormStats:
    all_rel = np.concatenate(
        [s.entity_rel[s.entity_mask.astype(bool)] for s in samples if s.entity_mask.any()]
        + [np.zeros((0, 3))], axis=0)
    if len(all_rel):
        rel_mean = all_rel.mean(axis=0)
        rel_std = all_rel.std(axis=0) + eps
    else:
        rel_mean = np.zeros(3)
        rel_std = np.ones(3)

    if samples:
        all_path = np.concatenate([s.path_target for s in samples], axis=0)
        path_std = all_path.std(axis=0) + eps
    else:
        path_std = np.ones(3)

    if samples:
        all_actions = np.concatenate([s.action_window for s in samples], axis=0)
        action_mean = all_actions.mean(axis=0)
        action_std = all_actions.std(axis=0) + eps
    else:
        action_mean = np.zeros(3)
        action_std = np.ones(3)

    return NormStats(rel_mean=rel_mean, rel_std=rel_std, path_std=path_std,
                      action_mean=action_mean, action_std=action_std)


# --------------------------------------------------------------------------
# Torch model
# --------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class ConsequenceModel(nn.Module):
        """GRU-over-actions + MLP-context trunk with 4 prediction heads."""

        def __init__(self, hidden: int = 64, ctx_hidden: int = 64,
                     n_domains: int = len(DOMAINS), e_slots: int = E, h_steps: int = H):
            super().__init__()
            self.h_steps = h_steps
            self.e_slots = e_slots
            ctx_dim = 3 + 2 + e_slots * 3 + e_slots + n_domains  # eef,grip,rel,mask,domain
            self.ctx_mlp = nn.Sequential(
                nn.Linear(ctx_dim, ctx_hidden), nn.ReLU(),
                nn.Linear(ctx_hidden, ctx_hidden), nn.ReLU(),
            )
            self.gru = nn.GRU(input_size=3 + ctx_hidden, hidden_size=hidden,
                               batch_first=True)
            self.increment_head = nn.Linear(hidden, 3)
            self.contact_head = nn.Linear(hidden + ctx_hidden, e_slots)
            self.holding_head = nn.Linear(hidden + ctx_hidden, 1)
            self.progress_head = nn.Linear(hidden + ctx_hidden, 1)

        def forward(self, eef, grip, entity_rel, entity_mask, domain_onehot, action_window):
            b = eef.shape[0]
            ctx_in = torch.cat([
                eef, grip,
                entity_rel.reshape(b, -1), entity_mask.reshape(b, -1),
                domain_onehot,
            ], dim=-1)
            ctx = self.ctx_mlp(ctx_in)  # (B, ctx_hidden)
            ctx_rep = ctx.unsqueeze(1).expand(-1, self.h_steps, -1)
            gru_in = torch.cat([action_window, ctx_rep], dim=-1)
            seq_out, h_n = self.gru(gru_in)  # seq_out: (B, H, hidden)
            increments = self.increment_head(seq_out)  # (B, H, 3)
            path = torch.cumsum(increments, dim=1)

            final_hidden = h_n[-1]  # (B, hidden)
            head_in = torch.cat([final_hidden, ctx], dim=-1)
            contact_logits = self.contact_head(head_in)
            holding_logit = self.holding_head(head_in).squeeze(-1)
            progress = self.progress_head(head_in).squeeze(-1)
            return path, contact_logits, holding_logit, progress

    def count_parameters(model: "ConsequenceModel") -> int:
        return sum(p.numel() for p in model.parameters())


    def samples_to_tensors(samples: Sequence[Sample], norm: NormStats, device="cpu"):
        domain_idx = {d: i for i, d in enumerate(DOMAINS)}
        n = len(samples)
        eef = np.zeros((n, 3))
        grip = np.zeros((n, 2))
        entity_rel = np.zeros((n, E, 3))
        entity_mask = np.zeros((n, E))
        domain_onehot = np.zeros((n, len(DOMAINS)))
        action_window = np.zeros((n, H, 3))
        path_target = np.zeros((n, H, 3))
        contact_target = np.zeros((n, E))
        contact_mask = np.zeros((n, E))
        holding_target = np.zeros((n,))
        progress_target = np.zeros((n,))
        progress_mask = np.zeros((n,))

        for k, s in enumerate(samples):
            eef[k] = s.eef  # raw metric eef (unnormalized by design, see NormStats docstring)
            grip[k] = s.grip
            entity_rel[k] = (s.entity_rel - norm.rel_mean) / norm.rel_std * s.entity_mask[:, None]
            entity_mask[k] = s.entity_mask
            domain_onehot[k, domain_idx[s.domain]] = 1.0
            action_window[k] = (s.action_window - norm.action_mean) / norm.action_std
            path_target[k] = s.path_target / norm.path_std  # zero-mean displacement, own scale
            contact_target[k] = s.contact_target
            contact_mask[k] = s.contact_mask
            holding_target[k] = s.holding_target
            progress_target[k] = s.progress_target
            progress_mask[k] = s.progress_mask

        t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
        return dict(
            eef=t(eef), grip=t(grip), entity_rel=t(entity_rel),
            entity_mask=t(entity_mask), domain_onehot=t(domain_onehot),
            action_window=t(action_window), path_target=t(path_target),
            contact_target=t(contact_target), contact_mask=t(contact_mask),
            holding_target=t(holding_target), progress_target=t(progress_target),
            progress_mask=t(progress_mask),
        )


    def compute_loss(model: "ConsequenceModel", batch: dict) -> Tuple["torch.Tensor", dict]:
        path, contact_logits, holding_logit, progress = model(
            batch["eef"], batch["grip"], batch["entity_rel"], batch["entity_mask"],
            batch["domain_onehot"], batch["action_window"])

        path_loss = F.mse_loss(path, batch["path_target"])

        c_loss_raw = F.binary_cross_entropy_with_logits(
            contact_logits, batch["contact_target"], reduction="none")
        c_mask = batch["contact_mask"]
        contact_loss = (c_loss_raw * c_mask).sum() / c_mask.sum().clamp(min=1.0)

        holding_loss = F.binary_cross_entropy_with_logits(
            holding_logit, batch["holding_target"])

        p_loss_raw = F.mse_loss(progress, batch["progress_target"], reduction="none")
        p_mask = batch["progress_mask"]
        progress_loss = (p_loss_raw * p_mask).sum() / p_mask.sum().clamp(min=1.0)

        total = path_loss + contact_loss + holding_loss + progress_loss
        parts = dict(path=float(path_loss), contact=float(contact_loss),
                     holding=float(holding_loss), progress=float(progress_loss))
        return total, parts


    def train_ensemble(
        train_samples: Sequence[Sample], held_samples: Sequence[Sample],
        norm: NormStats, out_dir: Path, n_members: int = 3,
        max_epochs: int = 200, patience: int = 10, lr: float = 1e-3,
        seeds: Optional[Sequence[int]] = None, batch_size: int = 256,
        device: str = "cpu",
    ) -> List[dict]:
        """Trains `n_members` independently-seeded models with early stopping
        on held-out loss. Returns per-member training history (list of dicts:
        {"seed", "epochs_run", "best_held_loss", "train_losses", "held_losses"})."""
        _require_torch()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        seeds = list(seeds) if seeds is not None else list(range(n_members))

        train_batch = samples_to_tensors(train_samples, norm, device=device)
        held_batch = samples_to_tensors(held_samples, norm, device=device) if held_samples else None

        histories = []
        for member_idx, seed in enumerate(seeds[:n_members]):
            torch.manual_seed(seed)
            model = ConsequenceModel()
            opt = torch.optim.Adam(model.parameters(), lr=lr)

            n = len(train_samples)
            best_held = float("inf")
            best_state = None
            epochs_since_improve = 0
            train_losses, held_losses = [], []

            rng = np.random.default_rng(seed)
            for epoch in range(max_epochs):
                order = rng.permutation(n) if n else np.array([], dtype=int)
                model.train()
                epoch_loss = 0.0
                n_batches = max(1, math.ceil(n / batch_size)) if n else 0
                for bstart in range(0, n, batch_size):
                    idx = order[bstart:bstart + batch_size]
                    batch = {k: v[idx] for k, v in train_batch.items()}
                    opt.zero_grad()
                    loss, _ = compute_loss(model, batch)
                    loss.backward()
                    opt.step()
                    epoch_loss += float(loss)
                epoch_loss = epoch_loss / max(n_batches, 1)
                train_losses.append(epoch_loss)

                model.eval()
                with torch.no_grad():
                    if held_batch is not None and len(held_samples):
                        held_loss, _ = compute_loss(model, held_batch)
                        held_loss = float(held_loss)
                    else:
                        held_loss = epoch_loss
                held_losses.append(held_loss)

                if held_loss < best_held - 1e-6:
                    best_held = held_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            ckpt_path = out_dir / f"consequence_model_member{member_idx}.pt"
            torch.save({"state_dict": model.state_dict(), "seed": seed}, ckpt_path)
            histories.append(dict(seed=seed, epochs_run=len(train_losses),
                                   best_held_loss=best_held,
                                   train_losses=train_losses, held_losses=held_losses))

        (out_dir / "norm_stats.json").write_text(json.dumps(norm.to_json(), indent=2))
        return histories


    def load_ensemble(out_dir: Path, n_members: int = 3) -> Tuple[List["ConsequenceModel"], NormStats]:
        _require_torch()
        out_dir = Path(out_dir)
        norm = NormStats.from_json(json.loads((out_dir / "norm_stats.json").read_text()))
        models = []
        for i in range(n_members):
            ckpt = torch.load(out_dir / f"consequence_model_member{i}.pt", map_location="cpu")
            model = ConsequenceModel()
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            models.append(model)
        return models, norm

else:  # pragma: no cover
    ConsequenceModel = None
    count_parameters = None
    samples_to_tensors = None
    compute_loss = None
    train_ensemble = None
    load_ensemble = None


# --------------------------------------------------------------------------
# Single-integrator baseline (no torch needed)
# --------------------------------------------------------------------------

def single_integrator_terminal(
    action_window: np.ndarray,
    translation_scale: float = BASELINE_TRANSLATION_SCALE,
    cmd_clip: float = BASELINE_CMD_CLIP,
) -> np.ndarray:
    """Calibrated single-integrator baseline terminal displacement (relative
    to the anchor eef): clip(action[:3], +-cmd_clip) * translation_scale,
    cumsum. Same rollout SHAPE as contextual_predictive_filter.filter_chunk,
    but the (2.0, 0.025) LS default here is calibrated from the exec-parity
    guidance-arm note in run_guided_libero_safety_eval.py:587-591, not from
    contextual_predictive_filter.py (whose defaults, translation_scale=0.05/
    max_translation_command=1.0, are instead the correct SL-domain
    calibration -- see BASELINE_CONFIG / baseline_params_for_domain(), which
    callers should use instead of these LS-only defaults when the sample's
    domain is known). `action_window` shape (H, 3) or (N, H, 3)."""
    clipped = np.clip(action_window, -cmd_clip, cmd_clip)
    deltas = clipped * translation_scale
    cum = np.cumsum(deltas, axis=-2)
    return cum[..., -1, :]


# --------------------------------------------------------------------------
# G1 metrics (no torch needed for the math itself; ensemble prediction does)
# --------------------------------------------------------------------------

def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Pooled AUC via the Mann-Whitney U / rank-sum identity. No sklearn
    dependency (sklearn is not guaranteed present in the `aegis` env)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    # average ranks for ties
    i = 0
    r = 1
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        r += (j - i + 1)
        i = j + 1
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")

    def _rank(x):
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(len(x), dtype=np.float64)
        sorted_x = x[order]
        i = 0
        r = 1
        while i < len(sorted_x):
            j = i
            while j + 1 < len(sorted_x) and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            avg_rank = (r + (r + (j - i))) / 2.0
            ranks[order[i:j + 1]] = avg_rank
            r += (j - i + 1)
            i = j + 1
        return ranks

    ra, rb = _rank(a), _rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _ok(v, thresh, ge=True):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    return v >= thresh if ge else v <= thresh


def _g1_metrics_for_subset(
    indices: np.ndarray,
    held_samples: Sequence[Sample],
    mean_path: np.ndarray,
    mean_contact_prob: np.ndarray,
    mean_progress: np.ndarray,
    norm: NormStats,
    err_ratio_thresh: float,
    auc_thresh: float,
    spearman_thresh: float,
) -> dict:
    """G1 metrics (a)/(b)/(c) + denominators + pass/unmeasurable for the
    subset of `held_samples` at `indices`. The single-integrator baseline
    for metric (a) uses EACH sample's OWN domain calibration
    (`baseline_params_for_domain`), so this is correct whether `indices`
    selects a single domain or is the full pooled set (reviewer fix round
    1: a uniform LS-calibrated baseline was wrong for SL-domain samples)."""
    n = len(indices)
    out = {"n_samples": n}
    if n == 0:
        out.update({
            "median_terminal_error_model": float("nan"),
            "median_terminal_error_baseline": float("nan"),
            "n_near_entity_samples": 0, "err_ratio": float("nan"),
            "n_contact_positives": 0, "n_slots_scored": 0, "auc": float("nan"),
            "n_progress_labeled": 0, "spearman": float("nan"),
            "unmeasurable": True, "pass": False,
        })
        return out

    sub_samples = [held_samples[i] for i in indices]
    sub_path = mean_path[indices]
    sub_contact_prob = mean_contact_prob[indices]
    sub_progress = mean_progress[indices]

    # --- (a) terminal-point error vs domain-calibrated single-integrator,
    # restricted to samples with any entity within NEAR_ENTITY_RADIUS.
    model_terminal = sub_path[:, -1, :] * norm.path_std  # back to meters
    near_mask = []
    baseline_terminal = np.zeros((n, 3))
    realized_terminal = np.zeros((n, 3))
    for k, s in enumerate(sub_samples):
        dists = np.linalg.norm(s.entity_rel * s.entity_mask[:, None], axis=-1)
        present = s.entity_mask.astype(bool)
        near = bool(present.any() and (dists[present] < NEAR_ENTITY_RADIUS).any())
        near_mask.append(near)
        scale, clip = baseline_params_for_domain(s.domain)
        baseline_terminal[k] = single_integrator_terminal(
            s.action_window, translation_scale=scale, cmd_clip=clip)
        realized_terminal[k] = s.path_target[-1]
    near_mask = np.array(near_mask)

    if near_mask.any():
        model_err = np.linalg.norm(
            model_terminal[near_mask] - realized_terminal[near_mask], axis=-1)
        baseline_err = np.linalg.norm(
            baseline_terminal[near_mask] - realized_terminal[near_mask], axis=-1)
        med_model = float(np.median(model_err))
        med_baseline = float(np.median(baseline_err))
        err_ratio = med_model / med_baseline if med_baseline > 0 else float("nan")
    else:
        med_model = med_baseline = err_ratio = float("nan")

    # --- (b) CONTACT AUC pooled over slots with any positive.
    c_target = np.stack([s.contact_target for s in sub_samples])
    c_mask = np.stack([s.contact_mask for s in sub_samples])
    n_contact_positives = int(c_target[c_mask.astype(bool)].sum())
    n_slots_scored = 0
    flat_scores, flat_labels = [], []
    for e in range(E):
        m = c_mask[:, e].astype(bool)
        if not m.any():
            continue
        if c_target[m, e].sum() == 0:
            continue  # no positives in this slot; skip (pooled AUC needs both classes)
        n_slots_scored += 1
        flat_scores.append(sub_contact_prob[m, e])
        flat_labels.append(c_target[m, e])
    if flat_scores:
        auc = _roc_auc(np.concatenate(flat_scores), np.concatenate(flat_labels))
    else:
        auc = float("nan")

    # --- (c) Spearman(progress pred, realized) over progress-labeled samples.
    p_mask = np.array([s.progress_mask for s in sub_samples], dtype=bool)
    p_target = np.array([s.progress_target for s in sub_samples])
    n_progress_labeled = int(p_mask.sum())
    if n_progress_labeled >= 2:
        spearman = _spearman(sub_progress[p_mask], p_target[p_mask])
    else:
        spearman = float("nan")

    err_ok = _ok(err_ratio, err_ratio_thresh, ge=False)
    auc_ok = _ok(auc, auc_thresh, ge=True)
    spearman_ok = _ok(spearman, spearman_thresh, ge=True)
    verdict = bool(err_ok and auc_ok and spearman_ok)
    unmeasurable = any(isinstance(v, float) and math.isnan(v)
                        for v in (err_ratio, auc, spearman))

    out.update({
        "median_terminal_error_model": med_model,
        "median_terminal_error_baseline": med_baseline,
        "n_near_entity_samples": int(near_mask.sum()),
        "n_contact_positives": n_contact_positives,
        "n_slots_scored": n_slots_scored,
        "n_progress_labeled": n_progress_labeled,
        "err_ratio": err_ratio,
        "auc": auc,
        "spearman": spearman,
        "unmeasurable": unmeasurable,
        "pass": verdict,
    })
    return out


def _g1_tag(metrics: dict) -> str:
    if metrics["pass"]:
        return "PASS"
    return "FAIL(unmeasurable)" if metrics.get("unmeasurable") else "FAIL"


def evaluate_g1(
    held_samples: Sequence[Sample],
    models: Sequence["ConsequenceModel"],
    norm: NormStats,
    err_ratio_thresh: float = 0.5,
    auc_thresh: float = 0.90,
    spearman_thresh: float = 0.6,
) -> dict:
    """Computes G1 offline metrics on held-out samples, PER-DOMAIN (each
    domain gated against its own thresholds using ITS OWN single-integrator
    baseline calibration -- LS and SL data-collection servers run at
    different (translation_scale, cmd_clip), see BASELINE_CONFIG), plus a
    pooled cut for information only. Returns the report dict (also the
    payload written to g1_report.json). Overall gate verdict ("pass") is
    True iff EVERY domain present in `held_samples` passes its own
    thresholds. Ensemble prediction = mean over members for path/progress,
    mean-of-probabilities for contact and holding."""
    _require_torch()
    thresh_kwargs = dict(err_ratio_thresh=err_ratio_thresh, auc_thresh=auc_thresh,
                          spearman_thresh=spearman_thresh)
    n = len(held_samples)
    report = {
        "n_samples": n,
        "domains_present": [],
        "per_domain": {},
        "pooled": _g1_metrics_for_subset(
            np.array([], dtype=int), held_samples, np.zeros((0, H, 3)),
            np.zeros((0, E)), np.zeros((0,)), norm, **thresh_kwargs),
        "err_ratio_thresh": err_ratio_thresh,
        "auc_thresh": auc_thresh,
        "spearman_thresh": spearman_thresh,
        "unmeasurable": True,
        "pass": False,
    }
    if n == 0:
        print("G1: FAIL(unmeasurable) (no held-out samples)")
        return report

    batch = samples_to_tensors(held_samples, norm)
    with torch.no_grad():
        paths, contact_probs, holding_probs, progresses = [], [], [], []
        for model in models:
            model.eval()
            path, contact_logits, holding_logit, progress = model(
                batch["eef"], batch["grip"], batch["entity_rel"], batch["entity_mask"],
                batch["domain_onehot"], batch["action_window"])
            paths.append(path.numpy())
            contact_probs.append(torch.sigmoid(contact_logits).numpy())
            holding_probs.append(torch.sigmoid(holding_logit).numpy())
            progresses.append(progress.numpy())
    mean_path = np.mean(paths, axis=0)          # (N, H, 3) normalized units
    mean_contact_prob = np.mean(contact_probs, axis=0)  # (N, E)
    mean_progress = np.mean(progresses, axis=0)  # (N,)

    domains_present = sorted({s.domain for s in held_samples})
    all_idx = np.arange(n)
    pooled = _g1_metrics_for_subset(
        all_idx, held_samples, mean_path, mean_contact_prob, mean_progress,
        norm, **thresh_kwargs)

    per_domain = {}
    for dom in domains_present:
        dom_idx = np.array([i for i, s in enumerate(held_samples) if s.domain == dom])
        per_domain[dom] = _g1_metrics_for_subset(
            dom_idx, held_samples, mean_path, mean_contact_prob, mean_progress,
            norm, **thresh_kwargs)

    overall_pass = bool(domains_present) and all(per_domain[d]["pass"] for d in domains_present)
    overall_unmeasurable = (not domains_present) or any(
        per_domain[d]["unmeasurable"] for d in domains_present)

    report.update({
        "domains_present": domains_present,
        "per_domain": per_domain,
        "pooled": pooled,
        "unmeasurable": overall_unmeasurable,
        "pass": overall_pass,
    })

    for dom in domains_present:
        m = per_domain[dom]
        print(f"G1[{dom}]: {_g1_tag(m)} "
              f"err_ratio={m['err_ratio']:.4f} auc={m['auc']:.4f} "
              f"spearman={m['spearman']:.4f} (n={m['n_samples']})")
    overall_tag = "FAIL(unmeasurable)" if (overall_unmeasurable and not overall_pass) \
        else ("PASS" if overall_pass else "FAIL")
    print(f"G1: {overall_tag} "
          f"pooled_err_ratio={pooled['err_ratio']:.4f} "
          f"pooled_auc={pooled['auc']:.4f} pooled_spearman={pooled['spearman']:.4f} "
          f"domains={domains_present}")
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _load_target_map(path: Optional[str]) -> Dict[int, str]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    return {int(k): v for k, v in raw.items()}


def _prepare_data(data_dir: Path, target_map_path: Optional[str], seed: int, out_dir: Optional[Path] = None):
    episodes = load_episodes(Path(data_dir))
    train_keys, held_keys = split_episodes(episodes, seed=seed)
    if out_dir is not None:
        save_split(Path(out_dir) / "split.json", train_keys, held_keys)
    train_keys_set, held_keys_set = set(train_keys), set(held_keys)
    train_eps = [e for e in episodes if e.key in train_keys_set]
    held_eps = [e for e in episodes if e.key in held_keys_set]
    target_map = _load_target_map(target_map_path)
    train_samples = build_samples(train_eps, target_map)
    held_samples = build_samples(held_eps, target_map)
    return train_samples, held_samples


def cmd_train(args):
    _require_torch()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_samples, held_samples = _prepare_data(
        Path(args.data_dir), args.target_map, args.seed, out_dir=out_dir)
    norm = compute_norm_stats(train_samples)
    histories = train_ensemble(
        train_samples, held_samples, norm, out_dir,
        n_members=args.n_members, max_epochs=args.max_epochs,
        patience=args.patience, lr=args.lr)
    (out_dir / "train_history.json").write_text(json.dumps(histories, indent=2))
    print(f"Trained {len(histories)} member(s) on {len(train_samples)} train / "
          f"{len(held_samples)} held-out samples -> {out_dir}")


def cmd_g1(args):
    _require_torch()
    model_dir = Path(args.model_dir)
    train_keys, held_keys = load_split(model_dir / "split.json")
    episodes = load_episodes(Path(args.data_dir))
    held_keys_set = set(held_keys)
    held_eps = [e for e in episodes if e.key in held_keys_set]
    target_map = _load_target_map(args.target_map)
    held_samples = build_samples(held_eps, target_map)
    models, norm = load_ensemble(model_dir, n_members=args.n_members)
    report = evaluate_g1(held_samples, models, norm)
    (model_dir / "g1_report.json").write_text(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data_dir", required=True)
    p_train.add_argument("--out_dir", required=True)
    p_train.add_argument("--target_map", default=None,
                          help="JSON file mapping task id -> target entity name")
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--n_members", type=int, default=3)
    p_train.add_argument("--max_epochs", type=int, default=200)
    p_train.add_argument("--patience", type=int, default=10)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.set_defaults(func=cmd_train)

    p_g1 = sub.add_parser("g1")
    p_g1.add_argument("--data_dir", required=True)
    p_g1.add_argument("--model_dir", required=True)
    p_g1.add_argument("--target_map", default=None)
    p_g1.add_argument("--n_members", type=int, default=3)
    p_g1.set_defaults(func=cmd_g1)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
