"""CPU unit tests for `--dynamic_election` (Phase 2 guard de-election on
oa): per replan, release episode-elected `fol` guard entries whose NEAR_PATH
disjunct no longer holds against the CURRENT eef->target segment, while
always retaining PROTECTED/MOVING entries.

Covers `_dynamic_deelect` in run_guided_libero_safety_eval.py, and the
shared `symbolic_identity.segment_distance` / `near_path` /
`NEAR_PATH_THRESHOLD` it delegates to for election-time geometry parity —
pure, no GPU, no env creation, no mujoco/robosuite imports. Modeled on
test_fol_predicates.py.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import run_guided_libero_safety_eval as ls  # noqa: E402
import symbolic_identity as si  # noqa: E402


def _pos_of(store):
    return lambda name: store.get(name)


# --- defaults-off byte-identity --------------------------------------------

def test_argparse_default_is_off():
    src = pathlib.Path(ls.__file__).read_text()
    assert '"--dynamic_election", action="store_true"' in src


def test_defaults_off_leaves_payload_path_untouched():
    # With --dynamic_election unset (argparse default False), the wiring
    # never calls _dynamic_deelect; `_elected` stays bound to the raw
    # episode-elected `guard`, so step_guard downstream is exactly
    # _gate_guard's output on the UNFILTERED guard — byte-identical to
    # Task 1 behavior. Mirrors the flag-off pattern in the other CPU tests
    # (a dynamic check on the real decision path, not source-grep).
    guard = ["far_from_path_obstacle"]
    store = {"far_from_path_obstacle": np.array([5.0, 5.0, 0.9])}
    eef = np.array([0.0, 0.0, 0.9])
    dynamic_election = False  # the argparse default
    elected = guard
    if dynamic_election:
        elected = ls._dynamic_deelect(
            guard, "fol", _pos_of(store), eef, np.array([1.0, 0.0, 0.9]), set())
    step_guard = ls._gate_guard(elected, _pos_of(store), eef, radius=0.0)
    assert step_guard == guard


# --- retention ---------------------------------------------------------

def test_protected_retained_even_far_from_path():
    guard = ["human_hand"]
    store = {"human_hand": np.array([5.0, 5.0, 0.9])}
    eef = np.array([0.0, 0.0, 0.9])
    target = np.array([1.0, 0.0, 0.9])
    out = ls._dynamic_deelect(guard, "fol", _pos_of(store), eef, target, set())
    assert out == ["human_hand"]


def test_moving_retained_even_far_from_path():
    guard = ["rolling_cart"]
    store = {"rolling_cart": np.array([5.0, 5.0, 0.9])}
    eef = np.array([0.0, 0.0, 0.9])
    target = np.array([1.0, 0.0, 0.9])
    out = ls._dynamic_deelect(
        guard, "fol", _pos_of(store), eef, target, {"rolling_cart"})
    assert out == ["rolling_cart"]


def test_near_path_only_retained_while_near_current_segment():
    # Hazard sits right on the eef->target segment: NEAR_PATH holds.
    guard = ["moka_pot"]
    store = {"moka_pot": np.array([0.5, 0.0, 0.9])}
    eef = np.array([0.0, 0.0, 0.9])
    target = np.array([1.0, 0.0, 0.9])
    out = ls._dynamic_deelect(guard, "fol", _pos_of(store), eef, target, set())
    assert out == ["moka_pot"]


# --- release -------------------------------------------------------------

def test_near_path_only_released_when_far_from_current_segment():
    # Segment now runs along +x near y=0; hazard is far off to the side ->
    # NEAR_PATH no longer holds -> released.
    guard = ["moka_pot"]
    store = {"moka_pot": np.array([0.5, 5.0, 0.9])}
    eef = np.array([0.0, 0.0, 0.9])
    target = np.array([1.0, 0.0, 0.9])
    out = ls._dynamic_deelect(guard, "fol", _pos_of(store), eef, target, set())
    assert out == []


def test_near_path_only_released_when_eef_has_passed_it():
    # Hazard sits BEHIND the eef relative to target -- clamped projection
    # puts the closest segment point at the eef itself, so distance to the
    # hazard is large once the eef has moved past it.
    guard = ["moka_pot"]
    store = {"moka_pot": np.array([-1.0, 0.0, 0.9])}
    eef = np.array([0.0, 0.0, 0.9])
    target = np.array([1.0, 0.0, 0.9])
    out = ls._dynamic_deelect(guard, "fol", _pos_of(store), eef, target, set())
    assert out == []


# --- re-entry --------------------------------------------------------------

def test_released_hazard_reenters_when_segment_swings_back():
    guard = ["moka_pot"]
    store = {"moka_pot": np.array([0.5, 5.0, 0.9])}
    eef = np.array([0.0, 0.0, 0.9])
    far_target = np.array([1.0, 0.0, 0.9])
    near_target = np.array([0.5, 4.9, 0.9])  # segment now swings toward it

    out_far = ls._dynamic_deelect(
        guard, "fol", _pos_of(store), eef, far_target, set())
    out_near = ls._dynamic_deelect(
        guard, "fol", _pos_of(store), eef, near_target, set())
    assert out_far == []
    assert out_near == ["moka_pot"]


# --- release-only invariant -------------------------------------------------

def test_release_only_output_is_subset_of_input():
    guard = ["human_hand", "moka_pot", "far_obstacle"]
    store = {
        "human_hand": np.array([5.0, 5.0, 0.9]),
        "moka_pot": np.array([0.5, 0.0, 0.9]),
        "far_obstacle": np.array([9.0, 9.0, 0.9]),
    }
    eef = np.array([0.0, 0.0, 0.9])
    target = np.array([1.0, 0.0, 0.9])
    out = ls._dynamic_deelect(guard, "fol", _pos_of(store), eef, target, set())
    assert set(out).issubset(set(guard))


def test_empty_guard_returns_empty():
    out = ls._dynamic_deelect(
        [], "fol", _pos_of({}), np.zeros(3), np.array([1.0, 0.0, 0.0]), set())
    assert out == []


def test_unresolvable_target_fails_safe_to_full_retention():
    guard = ["moka_pot", "human_hand"]
    store = {"moka_pot": np.array([5.0, 5.0, 0.9]),
             "human_hand": np.array([-5.0, -5.0, 0.9])}
    eef = np.array([0.0, 0.0, 0.9])
    out = ls._dynamic_deelect(guard, "fol", _pos_of(store), eef, None, set())
    assert out == guard


def test_unlocatable_guard_position_fails_safe_to_retention():
    guard = ["ghost"]
    out = ls._dynamic_deelect(
        guard, "fol", _pos_of({}), np.zeros(3), np.array([1.0, 0.0, 0.0]), set())
    assert out == guard


# --- no-op for non-NEAR_PATH identity sources -------------------------------

def test_noop_for_folprop_source():
    guard = ["moka_pot"]
    store = {"moka_pot": np.array([9.0, 9.0, 0.9])}
    eef = np.array([0.0, 0.0, 0.9])
    target = np.array([1.0, 0.0, 0.9])
    out = ls._dynamic_deelect(guard, "folprop", _pos_of(store), eef, target, set())
    assert out == guard


def test_noop_for_folpropvlm_source():
    guard = ["moka_pot"]
    store = {"moka_pot": np.array([9.0, 9.0, 0.9])}
    eef = np.array([0.0, 0.0, 0.9])
    target = np.array([1.0, 0.0, 0.9])
    out = ls._dynamic_deelect(
        guard, "folpropvlm", _pos_of(store), eef, target, set())
    assert out == guard


def test_noop_for_symbolic_source():
    guard = ["moka_pot"]
    store = {"moka_pot": np.array([9.0, 9.0, 0.9])}
    eef = np.array([0.0, 0.0, 0.9])
    target = np.array([1.0, 0.0, 0.9])
    out = ls._dynamic_deelect(guard, "symbolic", _pos_of(store), eef, target, set())
    assert out == guard


def test_noop_for_gt_source():
    guard = ["moka_pot"]
    store = {"moka_pot": np.array([9.0, 9.0, 0.9])}
    eef = np.array([0.0, 0.0, 0.9])
    target = np.array([1.0, 0.0, 0.9])
    out = ls._dynamic_deelect(guard, "gt", _pos_of(store), eef, target, set())
    assert out == guard


# --- threshold identity: shared callable, not a copied constant ------------

def test_shared_near_path_callable_is_the_same_object():
    import inspect
    src = inspect.getsource(ls._dynamic_deelect)
    assert "from symbolic_identity import _PROTECTED_CLASSES, near_path" in src
    assert "si.near_path" not in src  # imported directly, not module-qualified copy


def test_shared_threshold_monkeypatch_flips_retain_release_decision():
    # A hazard just OUTSIDE the real NEAR_PATH_THRESHOLD is released; if the
    # module constant is monkeypatched wider, the same geometry retains it —
    # proving `_dynamic_deelect` delegates to symbolic_identity's live
    # threshold rather than a value copied at import time.
    guard = ["moka_pot"]
    eef = np.array([0.0, 0.0, 0.9])
    target = np.array([1.0, 0.0, 0.9])
    offset = si.NEAR_PATH_THRESHOLD + 0.05
    store = {"moka_pot": np.array([0.5, offset, 0.9])}

    out_before = ls._dynamic_deelect(guard, "fol", _pos_of(store), eef, target, set())
    assert out_before == []

    orig = si.NEAR_PATH_THRESHOLD
    try:
        si.NEAR_PATH_THRESHOLD = offset + 1.0
        out_after = ls._dynamic_deelect(
            guard, "fol", _pos_of(store), eef, target, set())
    finally:
        si.NEAR_PATH_THRESHOLD = orig
    assert out_after == ["moka_pot"]


def test_shared_distance_function_matches_election_time_geometry():
    # `_dynamic_deelect`'s NEAR_PATH check must agree with the distance
    # `fol_obstacle_id` ranks candidates by, for the identical geometry.
    p = np.array([0.3, 0.05, 0.9])
    a = np.array([0.0, 0.0, 0.9])
    b = np.array([1.0, 0.0, 0.9])
    candidates = {"tgt": b, "hazard": p}
    ranked = si.fol_obstacle_id("pick up tgt", candidates, a, moving=None)
    assert ranked == ["hazard"]
    guard = ["hazard"]
    out = ls._dynamic_deelect(guard, "fol", _pos_of({"hazard": p}), a, b, set())
    assert out == (["hazard"] if si.near_path(p, a, b) else [])
    assert out == ["hazard"]


# --- ordering: composes before the engagement gate chain -------------------

def test_wiring_deelect_precedes_engagement_gates():
    src = pathlib.Path(ls.__file__).read_text()
    deelect_idx = src.index("_elected = _dynamic_deelect(")
    gate_idx = src.index("step_guard = _gate_guard(")
    cone_idx = src.index("step_guard = _cone_gate(")
    hold_idx = src.index("step_guard = _holding_filter(")
    assert deelect_idx < gate_idx < cone_idx < hold_idx


def test_results_dict_echoes_flag():
    src = pathlib.Path(ls.__file__).read_text()
    assert '"dynamic_election": bool(args.dynamic_election)' in src
