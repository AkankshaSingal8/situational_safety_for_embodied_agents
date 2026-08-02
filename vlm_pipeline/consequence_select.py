"""Consequence-model-based candidate selection for the guided server
(`--consequence_select`, Task 3 of the 2026-08-01 consequence-steering-code
plan).

Design doc: docs/superpowers/specs/2026-08-01-consequence-steering-design.md
Briefs:     .superpowers/sdd/2026-08-01-consequence-steering-code/task-2-brief.md
            .superpowers/sdd/2026-08-01-consequence-steering-code/task-3-brief.md

This module is imported by `openpi_guided/src/openpi/policies/guided_policy.py`
(JAX server process) and by the CPU test suite. It must be importable WITHOUT
jax, torch, or a GPU: only `numpy` is a hard top-level dependency (plus
`consequence_model`, which is itself torch-optional — see its module
docstring). `torch` is only imported inside functions that actually run the
ensemble (`load_ensemble_for_server`, `run_ensemble_select`), and those are
only called by the server when `--consequence_select MODEL_DIR` is passed.

--------------------------------------------------------------------------
Entity-slot contract (RESOLVED, fix round 1 -- was a documented v1 gap in
the initial version of this module; see task-3-report.md for history):

`consequence_model.py`'s binding entity-slot contract is
`sorted(union of entity NAMES seen in the episode)[:8]` -- alphabetical by
NAME, built from the named `entities` dict logged by `--log_transitions`.
The guided server's `guidance` request payload now carries the SAME named
`entities` dict (`{name: [x, y, z]}`, plus `gripper_qpos: [q0, q1]`),
attached by BOTH eval clients (`run_guided_libero_safety_eval.py`,
`run_guided_safelibero_pi05_eval.py`) via a single shared per-client helper
(`_resolve_entities`) that is also the source for their `--log_transitions`
logging -- the training-data source and the inference-time payload cannot
drift apart because they are literally the same function call. `build_features`
reproduces `consequence_model.build_samples`'s slot construction exactly:
`sorted(payload["entities"].keys())[:E]`, `rel = pos - eef`, `mask = 1` for
populated slots.

**Hard-fail contract.** When `payload["entities"]` is missing or empty (e.g.
an older client, or a request with `enabled=0`/no guard active), this is
NOT silently degraded to a positional-only approximation -- `build_features`
raises `EntitiesMissingError` and the caller (`GuidedPolicy._consequence_infer`)
catches it, logs a WARNING, and routes the query to the existing (legacy)
best-of-K selection unchanged, exactly like an empty-feasible-set fallback.
The old FIXED 4-pseudo-slot mapping (obstacle/obstacle2/target/dest keyed
off bare positions) is retained ONLY as `_build_features_legacy_pseudo_slots`
below for reference/back-compat -- it is dead code, never called by
`build_features` or any server path, and MUST NOT be reintroduced as a
silent fallback (that was the reviewer finding this fix round resolves).

**Mid-episode entity caveat.** The training-time slot order is the union of
entity names over the WHOLE episode (`EpisodeRecords.entity_order`), so an
entity that only becomes visible/relevant partway through an episode still
occupies the same slot for every sample in that episode. At inference the
payload's `entities` dict reflects only the CURRENT step's identification
candidate set (`cands`/`entity_view(obs)`), which in this codebase's current
clients is effectively static per episode (the candidate set is computed
once near episode start and not recomputed mid-episode), so this
distinction is currently moot in practice -- but a future client that
recomputes candidates mid-episode (e.g. a mover entering the workspace
partway through) could see slot 0 disagree between two requests in the same
episode where training saw one fixed union; this is called out here should
that ever become possible, not because it is possible today.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import consequence_model as _cm  # noqa: E402

E = _cm.E  # 8 entity slots
H = _cm.H  # 10-step action/path horizon
DOMAINS = _cm.DOMAINS  # ("LS", "SL")

class EntitiesMissingError(RuntimeError):
    """Raised by `build_features` when the guidance request payload has no
    (non-empty) `entities` dict. Fix round 1 (reviewer-mandated, binding):
    this is a HARD failure, not a silent degradation -- callers must catch
    it and route the query to the existing (legacy) best-of-K selection,
    logging a warning, exactly like an empty-feasible-set fallback. See
    `GuidedPolicy._consequence_infer`."""


# --------------------------------------------------------------------------
# DEAD CODE, retained only for reference (fix round 1): the original v1
# fixed 4-pseudo-slot mapping this module used before the guidance payload
# carried entity names. NOT called by `build_features` or any server path.
# Reintroducing this as a silent fallback for missing `entities` is exactly
# the pattern the reviewer's fix-round-1 finding prohibits -- see
# `EntitiesMissingError` and the module docstring's "Hard-fail contract".
# --------------------------------------------------------------------------
_LEGACY_PSEUDO_SLOTS = ("obstacle", "obstacle2", "target", "dest")
_LEGACY_SLOT_KEYS = {
    "obstacle": lambda payload: payload.get("obstacle_pos"),
    "obstacle2": lambda payload: (payload.get("obstacle2") or {}).get("pos"),
    "target": lambda payload: payload.get("target_pos"),
    "dest": lambda payload: payload.get("dest_pos"),
}


def _build_features_legacy_pseudo_slots(payload: dict, eef_pos: np.ndarray) -> tuple:
    """UNUSED (see module docstring / `EntitiesMissingError`). Kept only so
    the pre-fix-round-1 slot construction is visible for reference; returns
    (entity_rel (E,3), entity_mask (E,)) using the fixed obstacle/obstacle2/
    target/dest payload-key mapping. Not wired into `build_features`."""
    eef_pos = np.asarray(eef_pos, dtype=np.float64).reshape(3)
    entity_rel = np.zeros((E, 3), dtype=np.float64)
    entity_mask = np.zeros((E,), dtype=np.float64)
    for slot, name in enumerate(_LEGACY_PSEUDO_SLOTS):
        pos = _LEGACY_SLOT_KEYS[name](payload)
        if pos is not None:
            entity_rel[slot] = np.asarray(pos, dtype=np.float64).reshape(3) - eef_pos
            entity_mask[slot] = 1.0
    return entity_rel, entity_mask


# --------------------------------------------------------------------------
# Pure selection rule (no torch, no jax) -- the unit under test.
# --------------------------------------------------------------------------

@dataclass
class SelectionResult:
    feasible_mask: np.ndarray   # (K,) bool
    feasible_count: int
    chosen: Optional[int]       # None iff fallback
    fallback: bool
    p_mean: np.ndarray          # (K,) ensemble-mean P(contact any)
    p_std: np.ndarray           # (K,) ensemble std of P(contact any)
    progress_mean: np.ndarray   # (K,) ensemble-mean predicted progress

    def log_dict(self) -> dict:
        """The per-query debug line shape requested by the brief:
        `consequence_select: {"feasible": k, "chosen": i, "fallback": bool}`."""
        return {
            "feasible": int(self.feasible_count),
            "chosen": (int(self.chosen) if self.chosen is not None else None),
            "fallback": bool(self.fallback),
        }


def select_candidate(
    p_mean: np.ndarray,
    p_std: np.ndarray,
    progress_mean: np.ndarray,
    pessimism: float = 1.0,
    threshold: float = 0.10,
) -> SelectionResult:
    """Selection rule (design doc): feasible = p_mean + pessimism*p_std <=
    threshold; among feasible pick argmax(progress_mean); empty feasible ->
    fallback (chosen=None). Pure numpy, no ensemble/model dependency -- the
    ensemble-aggregation step (`aggregate_p_any`, `run_ensemble_select`)
    produces `p_mean`/`p_std`/`progress_mean` upstream of this function."""
    p_mean = np.asarray(p_mean, dtype=np.float64)
    p_std = np.asarray(p_std, dtype=np.float64)
    progress_mean = np.asarray(progress_mean, dtype=np.float64)
    if not (p_mean.shape == p_std.shape == progress_mean.shape):
        raise ValueError(
            f"shape mismatch: p_mean {p_mean.shape}, p_std {p_std.shape}, "
            f"progress_mean {progress_mean.shape}"
        )
    pessimistic_p = p_mean + pessimism * p_std
    feasible_mask = pessimistic_p <= threshold
    feasible_count = int(feasible_mask.sum())
    if feasible_count == 0:
        return SelectionResult(
            feasible_mask=feasible_mask, feasible_count=0, chosen=None,
            fallback=True, p_mean=p_mean, p_std=p_std, progress_mean=progress_mean,
        )
    masked_progress = np.where(feasible_mask, progress_mean, -np.inf)
    chosen = int(np.argmax(masked_progress))
    return SelectionResult(
        feasible_mask=feasible_mask, feasible_count=feasible_count, chosen=chosen,
        fallback=False, p_mean=p_mean, p_std=p_std, progress_mean=progress_mean,
    )


def aggregate_p_any(contact_prob: np.ndarray, entity_mask: np.ndarray) -> np.ndarray:
    """contact_prob: (K, E) per-slot sigmoid contact probabilities for ONE
    ensemble member. entity_mask: (E,) or (K, E) presence mask (1/0).
    Returns P(contact ANY present entity) = 1 - prod_{present slots}
    (1 - p_slot), i.e. treating per-slot contact events as independent --
    masked-out (absent) slots are excluded entirely (not treated as p=0,
    which would silently deflate p_any if a member is overconfident on an
    absent-entity slot; exclusion instead just drops the term)."""
    contact_prob = np.asarray(contact_prob, dtype=np.float64)
    entity_mask = np.asarray(entity_mask, dtype=np.float64)
    if entity_mask.ndim == 1:
        entity_mask = np.broadcast_to(entity_mask, contact_prob.shape)
    complement = np.where(entity_mask > 0, 1.0 - contact_prob, 1.0)
    return 1.0 - np.prod(complement, axis=-1)


# --------------------------------------------------------------------------
# Feature construction from the guidance request payload (pure numpy).
# --------------------------------------------------------------------------

def build_features(
    payload: dict,
    eef_pos: np.ndarray,
    disp_window: np.ndarray,
    domain: str,
    action_scale: float = 1.0,
    action_clip: float = 0.0,
) -> dict:
    """Builds the raw (unnormalized) per-candidate feature arrays consumed by
    `run_ensemble_select`.

    `payload`: the RAW guidance request dict (before `GuidanceParams`
    defaulting), so key presence reflects what the client actually sent.
    Requires `payload["entities"]` (a non-empty `{name: [x, y, z]}` dict,
    attached client-side by `_resolve_entities` -- see module docstring);
    raises `EntitiesMissingError` if absent/empty (hard-fail, fix round 1 --
    callers must catch this and fall back, not silently degrade).
    `eef_pos`: (3,) current end-effector position (same for every candidate
    -- it is the anchor state, not something denoising varies).
    `disp_window`: (K, >=H, 3) per-candidate per-step metric world-frame
    translation (`_prefix_acceptance`'s `disp`, from `pi0_guided.py`);
    truncated/used as the first H=10 steps to match the consequence model's
    fixed action horizon. This is the pre-`_plan_actions` "cmd * translation
    scale" displacement, NOT necessarily identical in scale/convention to
    the raw `action` field the model was trained on (which is the POST-
    `_plan_actions` env-step action -- see task-3-report.md "Action scale"
    concern). `action_scale`/`action_clip` let a caller correct for this if
    the client's `_plan_actions` mode is known (0 clip = no clip).

    Entity slots reproduce `consequence_model.build_samples`'s construction
    exactly: `sorted(payload["entities"].keys())[:E]`, `rel = pos - eef`,
    `mask = 1` for populated slots (remaining slots zero/mask=0) -- see
    `consequence_model.EpisodeRecords.entity_order`.

    `grip`: from `payload["gripper_qpos"]` when present (client-attached by
    the same fix-round-1 change); zeros otherwise (grip has no mask bit in
    the model, unlike entity slots, so a missing value is an unconditional
    degraded input rather than a hard failure -- documented per
    task-3-brief's "acceptable degradation for v1" allowance, unlike the
    entities case above).
    """
    if not payload.get("entities"):
        raise EntitiesMissingError(
            "guidance payload has no non-empty 'entities' dict -- "
            "consequence-select requires entity names to reproduce "
            "consequence_model's sorted(entities)[:E] training-time slot "
            "ordering. Caller must fall back to the legacy selection path."
        )
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")

    eef_pos = np.asarray(eef_pos, dtype=np.float64).reshape(3)
    disp_window = np.asarray(disp_window, dtype=np.float64)
    k = disp_window.shape[0]
    action_window = disp_window[:, :H, :]
    if action_window.shape[1] < H:
        pad = np.zeros((k, H - action_window.shape[1], 3))
        action_window = np.concatenate([action_window, pad], axis=1)
    action_window = action_window * action_scale
    if action_clip > 0:
        action_window = np.clip(action_window, -action_clip, action_clip)

    names = sorted(payload["entities"].keys())[:E]
    entity_rel = np.zeros((E, 3), dtype=np.float64)
    entity_mask = np.zeros((E,), dtype=np.float64)
    for slot, name in enumerate(names):
        pos = np.asarray(payload["entities"][name], dtype=np.float64).reshape(3)
        entity_rel[slot] = pos - eef_pos
        entity_mask[slot] = 1.0

    grip_raw = payload.get("gripper_qpos")
    grip = (np.asarray(grip_raw, dtype=np.float64).reshape(2)
            if grip_raw is not None else np.zeros(2, dtype=np.float64))

    return dict(
        eef=np.broadcast_to(eef_pos, (k, 3)).copy(),
        grip=np.broadcast_to(grip, (k, 2)).copy(),
        entity_rel=np.broadcast_to(entity_rel, (k, E, 3)).copy(),
        entity_mask=np.broadcast_to(entity_mask, (k, E)).copy(),
        domain=domain,
        action_window=action_window,
    )


# --------------------------------------------------------------------------
# Ensemble loading / inference (torch-dependent -- lazy import).
# --------------------------------------------------------------------------

def load_ensemble_for_server(model_dir: str, n_members: int = 3):
    """Loads the consequence-model ensemble for server-side CPU inference.

    Uses `consequence_model.load_ensemble` (Task 2's own loader) rather than
    replicating loading logic. Fails fast with a clear error identifying
    `model_dir` and the missing artifact on any load failure -- required at
    server start per the brief ("fail fast with a clear error if MODEL_DIR
    invalid"), since the bare `FileNotFoundError`/`RuntimeError` that
    `load_ensemble` raises on a missing `norm_stats.json` or checkpoint file
    does not itself name the offending directory.
    """
    model_dir = str(model_dir)
    try:
        models, norm = _cm.load_ensemble(model_dir, n_members=n_members)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"--consequence_select {model_dir!r}: failed to load consequence "
            f"model ensemble ({n_members} members expected) -- missing "
            f"artifact: {e}. Expected `norm_stats.json` and "
            f"`consequence_model_member{{0..{n_members - 1}}}.pt` under "
            f"{model_dir}."
        ) from e
    except Exception as e:  # noqa: BLE001 - re-raise with context, any failure mode
        raise RuntimeError(
            f"--consequence_select {model_dir!r}: failed to load consequence "
            f"model ensemble: {type(e).__name__}: {e}"
        ) from e
    for m in models:
        m.eval()
    return models, norm


def run_ensemble_select(
    models: Sequence,
    norm,
    features: dict,
    pessimism: float = 1.0,
    threshold: float = 0.10,
) -> SelectionResult:
    """Runs the loaded ensemble (CPU, eval mode, no_grad) on `features`
    (from `build_features`) and applies `select_candidate`. Torch is
    imported lazily here (only reached when `--consequence_select` is set)."""
    import torch  # noqa: PLC0415 - lazy: only when the flag is on

    # The consequence ensemble is a handful of tiny (~3-4e4 param) MLP/GRU
    # forward passes -- on a many-core node (e.g. 256-core HPC login/compute
    # nodes) torch's default thread pool launches one OMP thread per core for
    # every op, and thread-launch overhead alone can turn a sub-millisecond
    # forward into multiple SECONDS (measured: ~10s at K=8 on a 256-core
    # node with OMP_NUM_THREADS unset). Force single-threaded intra-op
    # execution for this call; it is by far the fastest setting for tensors
    # this small and keeps the ensemble from contending with the JAX server
    # process for CPU. Set on every call (cheap) rather than assuming caller
    # environment, since the server process may be shared with other work.
    torch.set_num_threads(1)

    eef = features["eef"]
    grip = features["grip"]
    entity_rel_raw = features["entity_rel"]
    entity_mask = features["entity_mask"]
    action_window_raw = features["action_window"]
    domain = features["domain"]

    domain_idx = {d: i for i, d in enumerate(DOMAINS)}
    k = eef.shape[0]
    domain_onehot = np.zeros((k, len(DOMAINS)), dtype=np.float64)
    domain_onehot[:, domain_idx[domain]] = 1.0

    # Normalize exactly as `consequence_model.samples_to_tensors` does.
    entity_rel = (entity_rel_raw - norm.rel_mean) / norm.rel_std * entity_mask[..., None]
    action_window = (action_window_raw - norm.action_mean) / norm.action_std

    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    eef_t, grip_t = t(eef), t(grip)
    entity_rel_t, entity_mask_t = t(entity_rel), t(entity_mask)
    domain_onehot_t, action_window_t = t(domain_onehot), t(action_window)

    p_any_members: List[np.ndarray] = []
    progress_members: List[np.ndarray] = []
    with torch.no_grad():
        for model in models:
            _path, contact_logits, _holding_logit, progress = model(
                eef_t, grip_t, entity_rel_t, entity_mask_t, domain_onehot_t, action_window_t
            )
            contact_prob = torch.sigmoid(contact_logits).numpy()
            p_any_members.append(aggregate_p_any(contact_prob, entity_mask))
            progress_members.append(progress.numpy())

    p_any_stack = np.stack(p_any_members, axis=0)      # (n_members, K)
    progress_stack = np.stack(progress_members, axis=0)  # (n_members, K)
    p_mean = p_any_stack.mean(axis=0)
    p_std = p_any_stack.std(axis=0)
    progress_mean = progress_stack.mean(axis=0)
    return select_candidate(p_mean, p_std, progress_mean, pessimism=pessimism, threshold=threshold)


if __name__ == "__main__":
    # Tiny structural self-check: feasibility + fallback, no model I/O.
    p_mean = np.array([0.05, 0.20, 0.02])
    p_std = np.array([0.02, 0.01, 0.30])  # candidate 2's pessimistic prob is high despite low mean
    progress = np.array([0.1, 0.9, 0.5])
    res = select_candidate(p_mean, p_std, progress)
    print("feasible:", res.feasible_mask, "chosen:", res.chosen, res.log_dict())

    all_infeasible = select_candidate(np.array([0.5, 0.6]), np.array([0.0, 0.0]), np.array([1.0, 2.0]))
    print("empty-feasible fallback:", all_infeasible.log_dict())
