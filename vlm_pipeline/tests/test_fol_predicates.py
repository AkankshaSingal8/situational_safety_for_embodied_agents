"""CPU unit tests for `--engage_cone` (APPROACHING(eef, hazard) velocity gate).

Covers `_cone_gate` in run_guided_libero_safety_eval.py — pure, no GPU, no
env creation, no mujoco/robosuite imports.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import run_guided_libero_safety_eval as ls  # noqa: E402

EEF = np.array([0.0, 0.0, 1.0])
HARD_R = ls.ENGAGE_CONE_HARD_R


def _pos_of(store):
    return lambda name: store.get(name)


def test_first_query_fallback_prev_none_all_engaged():
    # No velocity grounding yet -> every entry ENGAGED regardless of geometry
    # or position resolvability.
    store = {"far_obstacle": np.array([5.0, 5.0, 5.0])}
    guard = ["far_obstacle", "unlocatable"]
    out = ls._cone_gate(guard, _pos_of(store), EEF, None, cone_deg=10.0,
                         hard_radius=HARD_R)
    assert out == guard


def test_empty_guard_returns_empty():
    out = ls._cone_gate([], _pos_of({}), EEF, EEF, cone_deg=10.0,
                         hard_radius=HARD_R)
    assert out == []


def test_closing_within_cone_kept():
    # eef moving toward +x; obstacle straight ahead in +x, well outside hard radius.
    store = {"obs": np.array([1.0, 0.0, 1.0])}
    prev_eef = np.array([-0.1, 0.0, 1.0])
    eef = np.array([0.0, 0.0, 1.0])  # v = (0.1, 0, 0), toward obstacle
    out = ls._cone_gate(["obs"], _pos_of(store), eef, prev_eef, cone_deg=10.0,
                         hard_radius=HARD_R)
    assert out == ["obs"]


def test_receding_outside_cone_dropped():
    # eef moving in -x, obstacle is in +x direction -> angle ~180deg, dropped.
    store = {"obs": np.array([1.0, 0.0, 1.0])}
    prev_eef = np.array([0.1, 0.0, 1.0])
    eef = np.array([0.0, 0.0, 1.0])  # v = (-0.1, 0, 0), away from obstacle
    out = ls._cone_gate(["obs"], _pos_of(store), eef, prev_eef, cone_deg=10.0,
                         hard_radius=HARD_R)
    assert out == []


def test_tangent_geometry_boundary():
    # eef moving in +x; obstacle offset so the angle to it is ~45deg.
    store = {"obs": np.array([0.5, 0.5, 1.0])}
    prev_eef = np.array([-0.1, 0.0, 1.0])
    eef = np.array([0.0, 0.0, 1.0])  # v = (0.1, 0, 0); to_guard = (0.5,0.5,0) -> 45deg
    out_wide = ls._cone_gate(["obs"], _pos_of(store), eef, prev_eef,
                              cone_deg=50.0, hard_radius=HARD_R)
    out_narrow = ls._cone_gate(["obs"], _pos_of(store), eef, prev_eef,
                                cone_deg=40.0, hard_radius=HARD_R)
    assert out_wide == ["obs"]
    assert out_narrow == []


def test_hard_radius_overrides_heading():
    # Obstacle is very close (inside hard radius) but eef is moving AWAY from it.
    store = {"obs": np.array([0.05, 0.0, 1.0])}
    prev_eef = np.array([0.1, 0.0, 1.0])
    eef = np.array([0.0, 0.0, 1.0])  # v = (-0.1, 0, 0), receding
    dist = float(np.linalg.norm(store["obs"] - eef))
    assert dist <= HARD_R
    out = ls._cone_gate(["obs"], _pos_of(store), eef, prev_eef, cone_deg=1.0,
                         hard_radius=HARD_R)
    assert out == ["obs"]


def test_stationary_eef_outside_hard_radius_gates_out():
    store = {"obs": np.array([1.0, 0.0, 1.0])}
    prev_eef = np.array([0.0, 0.0, 1.0])
    eef = np.array([0.0, 0.0, 1.0])  # v = 0 -> speed <= 1e-4
    out = ls._cone_gate(["obs"], _pos_of(store), eef, prev_eef, cone_deg=90.0,
                         hard_radius=HARD_R)
    assert out == []


def test_unlocatable_entry_dropped_once_velocity_available():
    store = {}
    prev_eef = np.array([-0.1, 0.0, 1.0])
    eef = np.array([0.0, 0.0, 1.0])
    out = ls._cone_gate(["ghost"], _pos_of(store), eef, prev_eef,
                         cone_deg=90.0, hard_radius=HARD_R)
    assert out == []


def test_runner_up_entries_gated_independently():
    # primary closing, runner-up receding -> only primary survives.
    store = {
        "primary": np.array([1.0, 0.0, 1.0]),
        "runner_up": np.array([-1.0, 0.0, 1.0]),
    }
    prev_eef = np.array([-0.1, 0.0, 1.0])
    eef = np.array([0.0, 0.0, 1.0])  # v = (0.1, 0, 0)
    out = ls._cone_gate(["primary", "runner_up"], _pos_of(store), eef,
                         prev_eef, cone_deg=10.0, hard_radius=HARD_R)
    assert out == ["primary"]


def test_composition_with_gate_guard():
    # _gate_guard first (radius) filters out a far entry; _cone_gate then
    # gates the survivors by heading. Composition = sequential application.
    store = {
        "near_closing": np.array([0.1, 0.0, 1.0]),
        "far_closing": np.array([5.0, 0.0, 1.0]),
    }
    guard = ["near_closing", "far_closing"]
    eef = np.array([0.0, 0.0, 1.0])
    prev_eef = np.array([-0.1, 0.0, 1.0])
    radius_gated = ls._gate_guard(guard, _pos_of(store), eef, radius=0.5)
    assert radius_gated == ["near_closing"]
    cone_gated = ls._cone_gate(radius_gated, _pos_of(store), eef, prev_eef,
                                cone_deg=10.0, hard_radius=HARD_R)
    assert cone_gated == ["near_closing"]


def test_corridor_anchors_never_gated_by_construction():
    # _cone_gate only ever receives the guard list (primary/obstacle2); the
    # wiring never routes target_pos/dest_pos through it. Documented here as
    # a guard-list-only contract check: passing a corridor-anchor-styled
    # name through still obeys ordinary cone semantics (i.e. there is no
    # special-case exemption INSIDE the helper — exemption is structural in
    # the caller, which never calls this on target/dest).
    store = {"target_pos_like": np.array([1.0, 0.0, 1.0])}
    prev_eef = np.array([0.1, 0.0, 1.0])
    eef = np.array([0.0, 0.0, 1.0])  # receding
    out = ls._cone_gate(["target_pos_like"], _pos_of(store), eef, prev_eef,
                         cone_deg=10.0, hard_radius=HARD_R)
    assert out == []


def test_perpendicular_geometry_dropped():
    # v exactly perpendicular to (guard - eef) -> angle == 90deg.
    store = {"obs": np.array([0.0, 1.0, 1.0])}
    prev_eef = np.array([-0.1, 0.0, 1.0])
    eef = np.array([0.0, 0.0, 1.0])  # v = (0.1, 0, 0); to_guard = (0, 1, 0) -> 90deg
    out_excl = ls._cone_gate(["obs"], _pos_of(store), eef, prev_eef,
                              cone_deg=89.0, hard_radius=HARD_R)
    out_incl = ls._cone_gate(["obs"], _pos_of(store), eef, prev_eef,
                              cone_deg=90.0, hard_radius=HARD_R)
    assert out_excl == []
    assert out_incl == ["obs"]  # <= is inclusive at the boundary


def test_argparse_default_is_none_and_wiring_gate_present():
    # Real source-level check, not a hand-set local: --engage_cone must
    # default to None (off) in argparse, and the _cone_gate call site in
    # main()'s wiring must be conditioned on `args.engage_cone is not None`
    # so that leaving the flag off never invokes _cone_gate (byte-identical
    # payload path). A hand-set `= None` local would pass even if the real
    # argparse default were wrong, so this asserts against the source text.
    src = pathlib.Path(ls.__file__).read_text()
    assert '"--engage_cone", type=float, default=None' in src
    assert "if args.engage_cone is not None:" in src


def test_defaults_off_engage_cone_none_leaves_payload_path_untouched():
    # Mirrors the flag-off byte-identity contract: with --engage_cone unset
    # (argparse default None, verified above), the wiring skips _cone_gate
    # entirely and step_guard is exactly _gate_guard's output (default
    # engage_radius=0.0 -> guard unchanged too).
    guard = ["obstacle_a", "obstacle_b"]
    store = {"obstacle_a": np.array([0.2, 0.0, 1.0]),
             "obstacle_b": np.array([-0.2, 0.0, 1.0])}
    eef = np.array([0.0, 0.0, 1.0])
    engage_cone = None  # the argparse default, confirmed by the test above
    step_guard = ls._gate_guard(guard, _pos_of(store), eef, radius=0.0)
    if engage_cone is not None:
        step_guard = ls._cone_gate(step_guard, _pos_of(store), eef, None,
                                    engage_cone, HARD_R)
    assert step_guard == guard
