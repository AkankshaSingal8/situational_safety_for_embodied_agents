"""Pure-numpy pooling + selection helpers for Task 5 A/B steering
(server-side `--stall_resample`, `--progress_select`, spec 2026-08-12).

No jax/torch imports -- CPU-testable in an env with neither installed
(mirrors `consequence_select.py`'s split: the server-side JAX wiring lives
in `openpi_guided/src/openpi/policies/guided_policy.py`, which imports this
module lazily and only when a flag is actually on; this module itself is
plain numpy so it can be unit-tested without JAX/GPU).

Flag A (`--stall_resample N`): when the per-request payload signals
`stalled` AND the server has N > 0, the host loop in `guided_policy.py`
draws N ADDITIONAL fresh K-candidate batches (new noise seeds) and pools
them with the base batch before selection -- `pool_candidate_batches` does
the pooling here, `n_resample_batches` is the pure gate.

Flag B (`--progress_select`): among CERTIFIED (feasible) candidates in the
(possibly pooled) set, select argmax PROGRESS instead of the legacy
lexicographic rule, where PROGRESS is the kernel's own `acc["progress"]`
(`pi0_guided._prefix_acceptance`) -- executed-prefix net displacement
projected onto the unit vector toward `target_pos`, i.e. the SAME quantity
the existing `progress_weight` knob already adds into the legacy score as a
soft weighted term ("A1 directed progress", spec 2026-07-27). Flag B
promotes it to a hard argmax-among-CERTIFIED selector rather than inventing
a second, differently-scoped progress metric. `select_index` is the single
entry point both flags route through; it degrades to the exact legacy rule
(see `legacy_score`, a numpy mirror of `pi0_guided.guided_sample_actions`'s
own in-kernel score/argmax) whenever Flag B is off, no candidate is
feasible, or `target_pos` is missing/unresolved -- so it never crashes and
never changes behavior when the data required for progress-selection is
absent.
"""

import numpy as np

# Sentinel the server substitutes for an absent target/dest position
# (pi0_guided.py `GuidanceParams` / guided_policy.py `_make_params`'s `FAR`
# default). A raw request payload from the eval client never sends this
# value on purpose; checked here defensively in case a caller passes a
# GuidanceParams-derived value through instead of the raw payload field.
FAR_SENTINEL = np.array([100.0, 100.0, 100.0])


def n_resample_batches(stalled, stall_resample_n):
    """Flag A gate: number of ADDITIONAL K-candidate batches to draw beyond
    the base batch. Both the per-request `stalled` signal AND the server's
    `--stall_resample N > 0` must hold; either alone (or both off, the
    default) yields 0 -- no pooling, byte-identical to today."""
    return int(stall_resample_n) if (bool(stalled) and stall_resample_n > 0) else 0


def pool_candidate_batches(batches):
    """Concatenate a list of per-batch `dict[str, array]` (same keys,
    leading axis K) into one pooled dict, axis 0. A single-element list is a
    pass-through (new dict, same arrays) -- calling this when Flag A drew
    zero extra batches is a no-op on the values, only changing container
    identity."""
    if not batches:
        raise ValueError("pool_candidate_batches requires at least one batch")
    if len(batches) == 1:
        return dict(batches[0])
    keys = batches[0].keys()
    return {k: np.concatenate([np.asarray(b[k]) for b in batches], axis=0) for k in keys}


def legacy_score(feasible, min_margin, disp_norm, progress, tail_margin,
                  progress_weight, lookahead_weight):
    """Numpy mirror of `guided_sample_actions`'s own lexicographic
    scalarization over the K candidate axis (pi0_guided.py): executed-prefix
    feasibility dominates, then clearance margin, then net displacement,
    then the optional progress/lookahead terms. This is the selection rule
    used (a) whenever Flag B is off (so the pooled-set selection here stays
    consistent with the in-kernel single-batch selection) and (b) as Flag
    B's own fallback."""
    feasible = np.asarray(feasible, dtype=np.float64)
    return (1e3 * feasible
            + 10.0 * np.asarray(min_margin, dtype=np.float64)
            + 0.1 * np.asarray(disp_norm, dtype=np.float64)
            + progress_weight * np.asarray(progress, dtype=np.float64)
            + lookahead_weight * np.asarray(tail_margin, dtype=np.float64))


def target_is_valid(target_pos):
    """A request target anchor is usable iff present, a 3-vector, and not
    the FAR sentinel (unresolved anchor)."""
    if target_pos is None:
        return False
    target_pos = np.asarray(target_pos, dtype=np.float64)
    if target_pos.shape != (3,):
        return False
    return not np.allclose(target_pos, FAR_SENTINEL, atol=1e-3)


def select_index(feasible, min_margin, disp_norm, progress, tail_margin,
                  target_pos,
                  progress_weight, lookahead_weight, progress_select):
    """Selection rule over a (possibly Flag-A-pooled) candidate set. Pure:
    candidate arrays in, index out -- no jax, no I/O.

    `progress` is expected to be the kernel's own `acc["progress"]`
    (directed progress toward `target_pos` when a target is resolved;
    caller-supplied here only so the rule stays a pure function of arrays).

    `progress_select=False` -> the legacy lexicographic argmax (identical
    rule to the in-kernel selection; this is what Flag A alone reduces to
    when pooling extra candidates -- more seeds, same selection rule).

    `progress_select=True` -> argmax `progress` among CERTIFIED (feasible)
    candidates. Falls back to the legacy rule -- never crashes, never
    changes behavior when the data is absent -- if there is no feasible
    candidate or `target_pos` is missing/unresolved (in which case the
    kernel's own `progress` would silently be a disp_norm proxy, not true
    directed progress, so this gate is load-bearing, not just defensive)."""
    feasible = np.asarray(feasible, dtype=bool)
    fallback_idx = int(np.argmax(legacy_score(
        feasible, min_margin, disp_norm, progress, tail_margin,
        progress_weight, lookahead_weight)))
    if not progress_select or not feasible.any() or not target_is_valid(target_pos):
        return fallback_idx
    progress = np.asarray(progress, dtype=np.float64)
    prog_masked = np.where(feasible, progress, -np.inf)
    return int(np.argmax(prog_masked))
