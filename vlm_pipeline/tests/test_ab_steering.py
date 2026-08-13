"""CPU tests for Task 5 A/B steering (spec 2026-08-12):

  Flag A `--stall_resample N` (server): pool N additional K-candidate
  batches when the request signals `stalled`.
  Flag B `--progress_select` (server): argmax progress-toward-target
  selection among certified candidates.

Covers the pure numpy layer in `ab_steering.py` (candidate arrays in, index
out -- no jax/torch import, matches `test_consequence_select.py`'s split)
and the client-side pieces in `run_guided_libero_safety_eval.py`:
`_stall_signal` (the reused-detection gate) and the `--send_stall_signal`
CLI flag / guidance-payload byte-identity when off. `target_pos` is
verified to already ride the payload unconditionally via `_extend_guidance`
(Task 1/2's corridor-anchor work) — no new client field was needed for
Flag B.
"""

import collections
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ab_steering as ab  # noqa: E402
import run_guided_libero_safety_eval as ls  # noqa: E402


# ---------------------------------------------------------------------
# Flag A: n_resample_batches gate
# ---------------------------------------------------------------------

def test_n_resample_batches_off_by_default():
    assert ab.n_resample_batches(False, 0) == 0
    assert ab.n_resample_batches(True, 0) == 0  # N=0 -> off regardless of stalled
    assert ab.n_resample_batches(False, 3) == 0  # not stalled -> off regardless of N


def test_n_resample_batches_on_when_both_set():
    assert ab.n_resample_batches(True, 3) == 3
    assert ab.n_resample_batches(True, 1) == 1


# ---------------------------------------------------------------------
# pool_candidate_batches
# ---------------------------------------------------------------------

def test_pool_single_batch_is_passthrough():
    batch = {"min_margin": np.array([0.1, 0.2]), "feasible": np.array([True, False])}
    pooled = ab.pool_candidate_batches([batch])
    assert pooled is not batch
    np.testing.assert_array_equal(pooled["min_margin"], batch["min_margin"])
    np.testing.assert_array_equal(pooled["feasible"], batch["feasible"])


def test_pool_concatenates_along_axis_0():
    b1 = {"min_margin": np.array([0.1, 0.2]), "net_disp": np.zeros((2, 3))}
    b2 = {"min_margin": np.array([0.3]), "net_disp": np.ones((1, 3))}
    pooled = ab.pool_candidate_batches([b1, b2])
    np.testing.assert_array_equal(pooled["min_margin"], np.array([0.1, 0.2, 0.3]))
    assert pooled["net_disp"].shape == (3, 3)


def test_pool_requires_nonempty():
    with pytest.raises(ValueError):
        ab.pool_candidate_batches([])


# ---------------------------------------------------------------------
# legacy_score / select_index defaults-off byte-identity
# ---------------------------------------------------------------------

def test_legacy_score_matches_hand_computation():
    feasible = np.array([True, False, True])
    min_margin = np.array([0.01, 0.5, 0.02])
    disp_norm = np.array([1.0, 2.0, 3.0])
    progress = np.array([0.0, 0.0, 0.0])
    tail_margin = np.array([0.0, 0.0, 0.0])
    score = ab.legacy_score(feasible, min_margin, disp_norm, progress, tail_margin,
                              progress_weight=0.0, lookahead_weight=0.0)
    expected = np.array([
        1e3 * 1 + 10.0 * 0.01 + 0.1 * 1.0,
        1e3 * 0 + 10.0 * 0.5 + 0.1 * 2.0,
        1e3 * 1 + 10.0 * 0.02 + 0.1 * 3.0,
    ])
    np.testing.assert_allclose(score, expected)


def test_select_index_progress_select_off_uses_legacy_argmax():
    feasible = np.array([True, True, False])
    min_margin = np.array([0.01, 0.05, 0.9])  # candidate 1 has higher margin
    disp_norm = np.zeros(3)
    progress = np.zeros(3)
    tail_margin = np.zeros(3)
    idx = ab.select_index(
        feasible, min_margin, disp_norm, progress, tail_margin,
        target_pos=np.array([1.0, 0.0, 0.0]),
        progress_weight=0.0, lookahead_weight=0.0, progress_select=False,
    )
    assert idx == 1  # highest legacy score among feasible (margin dominates after feasibility)


# ---------------------------------------------------------------------
# target validity
# ---------------------------------------------------------------------

def test_target_is_valid():
    assert not ab.target_is_valid(None)
    assert not ab.target_is_valid(ab.FAR_SENTINEL)
    assert not ab.target_is_valid(np.array([1.0, 2.0]))  # wrong shape
    assert ab.target_is_valid(np.array([0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------
# select_index with progress_select on: main behavior + fallbacks.
# `progress` here plays the role of the kernel's own `acc["progress"]`
# (pi0_guided._prefix_acceptance: prefix net displacement projected onto
# the unit vector toward target_pos) -- select_index treats it as an
# opaque per-candidate score, so these values are hand-planted directly
# rather than re-derived from eef_pos/net_disp.
# ---------------------------------------------------------------------

def test_progress_select_picks_argmax_progress_among_feasible():
    feasible = np.array([True, True, True])
    min_margin = np.array([0.5, 0.5, 0.5])  # tie on legacy terms
    disp_norm = np.zeros(3)
    progress = np.array([0.1, 0.9, 0.5])  # candidate 1 has best progress
    tail_margin = np.zeros(3)
    idx = ab.select_index(
        feasible, min_margin, disp_norm, progress, tail_margin,
        target_pos=np.array([1.0, 0.0, 0.0]),
        progress_weight=0.0, lookahead_weight=0.0, progress_select=True,
    )
    assert idx == 1


def test_progress_select_ignores_infeasible_candidates():
    feasible = np.array([False, True, False])
    min_margin = np.zeros(3)
    disp_norm = np.zeros(3)
    # Candidate 0 has the best raw progress but is INFEASIBLE -> excluded.
    progress = np.array([10.0, 0.2, 5.0])
    tail_margin = np.zeros(3)
    idx = ab.select_index(
        feasible, min_margin, disp_norm, progress, tail_margin,
        target_pos=np.array([1.0, 0.0, 0.0]),
        progress_weight=0.0, lookahead_weight=0.0, progress_select=True,
    )
    assert idx == 1


def test_progress_select_falls_back_when_no_feasible_candidate():
    feasible = np.array([False, False])
    min_margin = np.array([0.1, 0.2])  # candidate 1 wins the legacy fallback
    disp_norm = np.zeros(2)
    progress = np.zeros(2)
    tail_margin = np.zeros(2)
    idx = ab.select_index(
        feasible, min_margin, disp_norm, progress, tail_margin,
        target_pos=np.array([1.0, 0.0, 0.0]),
        progress_weight=0.0, lookahead_weight=0.0, progress_select=True,
    )
    assert idx == 1  # legacy argmax fallback, never crashes


def test_progress_select_falls_back_when_target_missing():
    feasible = np.array([True, True])
    min_margin = np.array([0.1, 0.2])  # candidate 1 wins the legacy fallback
    disp_norm = np.zeros(2)
    progress = np.array([10.0, 0.0])  # candidate 0 would win on progress
    tail_margin = np.zeros(2)
    idx = ab.select_index(
        feasible, min_margin, disp_norm, progress, tail_margin,
        target_pos=None,
        progress_weight=0.0, lookahead_weight=0.0, progress_select=True,
    )
    assert idx == 1  # target missing -> legacy fallback, not the progress winner


# ---------------------------------------------------------------------
# Compose: pooled candidates (Flag A) + progress selection (Flag B)
# ---------------------------------------------------------------------

def test_compose_pooled_candidates_progress_selects_planted_argmax():
    # Base batch: 2 candidates, both mediocre progress.
    base = {
        "feasible": np.array([True, True]),
        "min_margin": np.array([0.1, 0.1]),
        "disp_norm": np.zeros(2),
        "progress": np.array([0.1, 0.05]),
        "tail_margin": np.zeros(2),
        "net_disp": np.array([[0.1, 0.0, 0.0], [0.05, 0.0, 0.0]]),  # logged only
    }
    # Resampled batch (Flag A): 2 fresh candidates, one is the planted best.
    extra = {
        "feasible": np.array([True, True]),
        "min_margin": np.array([0.1, 0.1]),
        "disp_norm": np.zeros(2),
        "progress": np.array([0.02, 0.95]),  # planted argmax at pooled idx 3
        "tail_margin": np.zeros(2),
        "net_disp": np.array([[0.02, 0.0, 0.0], [0.95, 0.0, 0.0]]),
    }
    n_extra = ab.n_resample_batches(stalled=True, stall_resample_n=1)
    assert n_extra == 1
    pooled = ab.pool_candidate_batches([base, extra])
    assert pooled["min_margin"].shape == (4,)

    idx = ab.select_index(
        pooled["feasible"], pooled["min_margin"], pooled["disp_norm"],
        pooled["progress"], pooled["tail_margin"],
        target_pos=np.array([1.0, 0.0, 0.0]),
        progress_weight=0.0, lookahead_weight=0.0, progress_select=True,
    )
    assert idx == 3  # the planted best-progress candidate in the resampled batch


def test_compose_no_resample_is_single_batch_selection():
    base = {
        "feasible": np.array([True, True]),
        "min_margin": np.array([0.1, 0.1]),
        "disp_norm": np.zeros(2),
        "progress": np.array([0.9, 0.1]),
        "tail_margin": np.zeros(2),
        "net_disp": np.array([[0.9, 0.0, 0.0], [0.1, 0.0, 0.0]]),
    }
    n_extra = ab.n_resample_batches(stalled=False, stall_resample_n=3)  # not stalled -> 0
    assert n_extra == 0
    pooled = ab.pool_candidate_batches([base])
    idx = ab.select_index(
        pooled["feasible"], pooled["min_margin"], pooled["disp_norm"],
        pooled["progress"], pooled["tail_margin"],
        target_pos=np.array([1.0, 0.0, 0.0]),
        progress_weight=0.0, lookahead_weight=0.0, progress_select=True,
    )
    assert idx == 0


# ---------------------------------------------------------------------
# Client side: _stall_signal (reused detection, not the +Z response) and
# --send_stall_signal / guidance-payload byte-identity when off.
# ---------------------------------------------------------------------

def _full_hist(positions, maxlen=5):
    d = collections.deque(maxlen=maxlen)
    for p in positions:
        d.append(np.asarray(p, dtype=np.float64))
    return d


def test_stall_signal_off_by_default():
    hist = _full_hist([[0, 0, 0]] * 5)
    assert ls._stall_signal(False, hist, 0.01, ["hazard"]) is False


def test_stall_signal_requires_guard_engaged():
    # Window is full and stalled (all identical positions), but no guard.
    hist = _full_hist([[0, 0, 0]] * 5)
    assert ls._stall_signal(True, hist, 0.01, []) is False


def test_stall_signal_fires_when_stalled_and_guarded():
    hist = _full_hist([[0, 0, 0]] * 5)  # net displacement 0 < d_stall
    assert ls._stall_signal(True, hist, 0.01, ["hazard"]) is True


def test_stall_signal_false_when_window_not_full():
    hist = _full_hist([[0, 0, 0]] * 2, maxlen=5)  # only 2/5 entries
    assert ls._stall_signal(True, hist, 0.01, ["hazard"]) is False


def test_stall_signal_false_when_moving():
    positions = [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0], [0.4, 0, 0]]
    hist = _full_hist(positions)
    assert ls._stall_signal(True, hist, 0.01, ["hazard"]) is False


def test_send_stall_signal_argparse_default_is_off():
    parser_src = pathlib.Path(ls.__file__).read_text()
    assert '"--send_stall_signal"' in parser_src
    assert 'ap.add_argument("--send_stall_signal", action="store_true"' in parser_src


# ---------------------------------------------------------------------
# target_pos already rides the payload unconditionally (Task 1/2's
# _extend_guidance) -- Flag B needed no new client field. Drive the
# existing payload-construction function directly.
# ---------------------------------------------------------------------

def test_extend_guidance_already_carries_target_pos():
    payload = {"enabled": 1.0}

    def pos_of(name):
        return {"guard_obj": np.array([0.2, 0.2, 0.9]),
                "the_target": np.array([0.5, 0.1, 0.85])}.get(name)

    out = ls._extend_guidance(payload, ["guard_obj"], pos_of, "the_target", None)
    assert "target_pos" in out
    np.testing.assert_allclose(out["target_pos"], np.array([0.5, 0.1, 0.85], dtype=np.float32))


def test_extend_guidance_omits_target_pos_when_unresolved():
    payload = {"enabled": 1.0}

    def pos_of(name):
        return None

    out = ls._extend_guidance(payload, [], pos_of, None, None)
    assert "target_pos" not in out
    assert "dest_pos" not in out


# ---------------------------------------------------------------------
# Server wiring: source-level checks (this env has no JAX/GPU guarantee,
# same split as test_consequence_select.py's source-level checks for
# guided_policy.py/pi0_guided.py).
# ---------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_guided_policy_source_defaults_off():
    src = (_REPO_ROOT / "openpi_guided" / "src" / "openpi" / "policies"
           / "guided_policy.py").read_text()
    assert "stall_resample: int = 0" in src
    assert "progress_select: bool = False" in src
    assert "self._ab = None" in src
    # infer() must only route to _ab_infer when a flag is actually on
    # (elif self._ab is not None), so the off path falls through to the
    # pre-existing self._sample_actions call exactly as before this change.
    assert "elif self._ab is not None:" in src


def test_guided_policy_source_ab_infer_backstops_to_legacy_output():
    src = (_REPO_ROOT / "openpi_guided" / "src" / "openpi" / "policies"
           / "guided_policy.py").read_text()
    fn_start = src.index("def _ab_infer(")
    fn_end = src.index("\n    def ", fn_start + 1)
    body = src[fn_start:fn_end]
    # Non-stalled A-solo / both-off path returns the exact legacy 5-tuple's
    # selected/diag verbatim -- not a numpy re-derivation.
    assert "if n_extra == 0 and not cfg.progress_select:" in body
    assert "return selected, diag" in body


def test_guided_policy_source_guards_num_candidates_and_incompatible_flags():
    src = (_REPO_ROOT / "openpi_guided" / "src" / "openpi" / "policies"
           / "guided_policy.py").read_text()
    assert "if config.stall_resample > 0 or config.progress_select:" in src
    assert "if config.num_candidates <= 1:" in src
    assert "if config.consequence_select:" in src  # mutual-exclusion guard


def test_serve_policy_guided_source_exposes_ab_flags():
    src = (_REPO_ROOT / "openpi_guided" / "scripts" / "serve_policy_guided.py").read_text()
    assert "stall_resample: int = 0" in src
    assert "progress_select: bool = False" in src


def test_no_obstacle_name_substring_in_ab_steering():
    src = (_REPO_ROOT / "vlm_pipeline" / "ab_steering.py").read_text()
    assert "_obstacle_" not in src
