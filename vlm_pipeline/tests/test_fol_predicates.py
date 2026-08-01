"""CPU unit tests for `--engage_cone` (APPROACHING(eef, hazard) velocity gate),
`--holding_disengage` (HOLDING(eef) gripper-state suppression), and
`--mover_lead` (MOVER_LEAD(hazard) predictive barrier anchor).

Covers `_cone_gate`, `_is_holding`, `_holding_filter`, `_lead_positions` in
run_guided_libero_safety_eval.py — pure, no GPU, no env creation, no
mujoco/robosuite imports.
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


# --- `--holding_disengage` (HOLDING(eef) gripper-state suppression) -------

WIDTH_MIN = ls.HOLDING_WIDTH_MIN
WIDTH_MAX = ls.HOLDING_WIDTH_MAX


def test_is_holding_width_below_min_edge_not_holding():
    # Near-zero width (closed on nothing) is excluded even with streak met.
    assert ls._is_holding(WIDTH_MIN, streak=5) is False


def test_is_holding_width_at_min_edge_exclusive_not_holding():
    assert ls._is_holding(WIDTH_MIN, streak=5) is False


def test_is_holding_width_at_max_edge_exclusive_not_holding():
    assert ls._is_holding(WIDTH_MAX, streak=5) is False


def test_is_holding_width_above_max_edge_not_holding():
    # Fully open gripper.
    assert ls._is_holding(0.08, streak=5) is False


def test_is_holding_width_mid_band_holding_with_streak():
    mid = (WIDTH_MIN + WIDTH_MAX) / 2.0
    assert ls._is_holding(mid, streak=2) is True


def test_is_holding_streak_requirement_below_two_not_holding():
    mid = (WIDTH_MIN + WIDTH_MAX) / 2.0
    assert ls._is_holding(mid, streak=1) is False
    assert ls._is_holding(mid, streak=0) is False


def test_is_holding_streak_exactly_two_is_holding():
    mid = (WIDTH_MIN + WIDTH_MAX) / 2.0
    assert ls._is_holding(mid, streak=2) is True
    assert ls._is_holding(mid, streak=3) is True


def test_holding_filter_static_suppressed_mover_kept():
    guard = ["static_obj", "moving_hand"]
    movers = {"moving_hand"}
    out = ls._holding_filter(guard, movers, is_holding=True)
    assert out == ["moving_hand"]


def test_holding_filter_not_holding_passthrough_unchanged():
    guard = ["static_obj", "moving_hand"]
    movers = {"moving_hand"}
    out = ls._holding_filter(guard, movers, is_holding=False)
    assert out == guard


def test_holding_filter_empty_guard_returns_empty():
    out = ls._holding_filter([], {"moving_hand"}, is_holding=True)
    assert out == []


def test_holding_filter_missing_movers_metadata_no_suppression():
    # movers is None -> metadata unavailable -> conservative: suppress nothing.
    guard = ["static_obj", "another_static_obj"]
    out = ls._holding_filter(guard, None, is_holding=True)
    assert out == guard


def test_holding_filter_all_static_all_suppressed_when_movers_known_empty():
    # movers metadata IS available but empty (nothing observed moving this
    # episode) -> every entry is static -> all suppressed while holding.
    guard = ["static_obj_a", "static_obj_b"]
    out = ls._holding_filter(guard, set(), is_holding=True)
    assert out == []


def test_holding_filter_never_suppresses_mover_or_hand_entries():
    guard = ["hand_1", "static_obj"]
    movers = {"hand_1"}
    out = ls._holding_filter(guard, movers, is_holding=True)
    assert "hand_1" in out
    assert "static_obj" not in out


def test_argparse_holding_disengage_default_off_and_wiring_present():
    src = pathlib.Path(ls.__file__).read_text()
    assert '"--holding_disengage", action="store_true"' in src
    assert "if args.holding_disengage:" in src
    # Order requirement: _gate_guard -> _cone_gate -> _holding_filter.
    gate_idx = src.index("step_guard = _gate_guard(")
    cone_idx = src.index("step_guard = _cone_gate(")
    hold_idx = src.index("step_guard = _holding_filter(")
    assert gate_idx < cone_idx < hold_idx


def test_defaults_off_holding_disengage_leaves_payload_path_untouched():
    # With --holding_disengage unset (argparse default False), the wiring
    # never calls _holding_filter, so step_guard is exactly whatever
    # _gate_guard/_cone_gate produced — byte-identical to Task 1 behavior.
    guard = ["static_obj", "moving_hand"]
    store = {"static_obj": np.array([0.2, 0.0, 1.0]),
             "moving_hand": np.array([-0.2, 0.0, 1.0])}
    eef = np.array([0.0, 0.0, 1.0])
    holding_disengage = False  # the argparse default
    step_guard = ls._gate_guard(guard, _pos_of(store), eef, radius=0.0)
    if holding_disengage:
        step_guard = ls._holding_filter(step_guard, {"moving_hand"}, True)
    assert step_guard == guard


# --- `--mover_lead` (MOVER_LEAD(hazard) predictive barrier anchor) --------

LEAD_CAP = ls.MOVER_LEAD_CAP


def test_lead_positions_extrapolation_math_exact():
    # mover with a small, sub-cap delta: anchor = pos + (pos-prev)*lead_steps.
    pos_of = {"hand": np.array([0.20, 0.0, 1.0])}
    prev_pos_of = {"hand": np.array([0.19, 0.0, 1.0])}  # delta = (0.01, 0, 0)
    out = ls._lead_positions(pos_of, prev_pos_of, {"hand"}, lead_steps=3.0,
                              cap=LEAD_CAP)
    expected = np.array([0.23, 0.0, 1.0])  # 0.20 + 0.01*3, well under cap (0.03 < 0.10)
    assert set(out.keys()) == {"hand"}
    np.testing.assert_allclose(out["hand"], expected)


def test_lead_positions_cap_saturation_direction_preserved():
    # delta*lead_steps norm exceeds cap -> clamped to `cap` in the same direction.
    pos_of = {"hand": np.array([0.0, 0.0, 1.0])}
    prev_pos_of = {"hand": np.array([-0.1, 0.0, 1.0])}  # delta = (0.1, 0, 0)
    out = ls._lead_positions(pos_of, prev_pos_of, {"hand"}, lead_steps=5.0,
                              cap=LEAD_CAP)
    # Uncapped displacement would be (0.5, 0, 0), norm 0.5 >> cap 0.10.
    disp = out["hand"] - pos_of["hand"]
    assert np.isclose(float(np.linalg.norm(disp)), LEAD_CAP)
    np.testing.assert_allclose(disp / np.linalg.norm(disp), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(out["hand"], pos_of["hand"] + np.array([LEAD_CAP, 0.0, 0.0]))


def test_lead_positions_first_query_passthrough():
    # No previous sample for this entity yet -> observed position unchanged,
    # even though it IS a mover.
    pos_of = {"hand": np.array([0.2, 0.0, 1.0])}
    prev_pos_of = {}  # entity absent -> first query
    out = ls._lead_positions(pos_of, prev_pos_of, {"hand"}, lead_steps=10.0,
                              cap=LEAD_CAP)
    np.testing.assert_allclose(out["hand"], pos_of["hand"])


def test_lead_positions_static_entry_untouched():
    # name not in movers -> unchanged regardless of prev/lead_steps.
    pos_of = {"table": np.array([0.5, 0.5, 0.8])}
    prev_pos_of = {"table": np.array([0.1, 0.1, 0.8])}  # large delta, ignored
    out = ls._lead_positions(pos_of, prev_pos_of, {"hand"}, lead_steps=10.0,
                              cap=LEAD_CAP)
    np.testing.assert_allclose(out["table"], pos_of["table"])


def test_lead_positions_missing_movers_metadata_all_untouched():
    # movers is None (episode movers metadata unavailable) -> conservative:
    # every entry passes through unchanged, even ones with a real prev sample.
    pos_of = {"hand": np.array([0.2, 0.0, 1.0])}
    prev_pos_of = {"hand": np.array([0.0, 0.0, 1.0])}
    out = ls._lead_positions(pos_of, prev_pos_of, None, lead_steps=10.0,
                              cap=LEAD_CAP)
    np.testing.assert_allclose(out["hand"], pos_of["hand"])


def test_lead_positions_multiple_entries_independent():
    pos_of = {
        "mover_with_prev": np.array([1.0, 0.0, 1.0]),
        "mover_first_query": np.array([2.0, 0.0, 1.0]),
        "static": np.array([3.0, 0.0, 1.0]),
    }
    prev_pos_of = {
        "mover_with_prev": np.array([0.9, 0.0, 1.0]),  # delta (0.1,0,0)
        "static": np.array([2.9, 0.0, 1.0]),
    }
    movers = {"mover_with_prev", "mover_first_query"}
    out = ls._lead_positions(pos_of, prev_pos_of, movers, lead_steps=0.5,
                              cap=LEAD_CAP)
    np.testing.assert_allclose(out["mover_with_prev"], [1.05, 0.0, 1.0])
    np.testing.assert_allclose(out["mover_first_query"], pos_of["mover_first_query"])
    np.testing.assert_allclose(out["static"], pos_of["static"])


def test_argparse_mover_lead_default_none_and_wiring_present():
    src = pathlib.Path(ls.__file__).read_text()
    assert '"--mover_lead", type=float, default=None' in src
    assert "if args.mover_lead is not None:" in src
    assert "_lead_pos = _lead_anchor_pos = _guard_pos" in src
    # Wiring composes AFTER _gate_guard -> _cone_gate -> _holding_filter.
    hold_idx = src.index("step_guard = _holding_filter(")
    lead_idx = src.index("if args.mover_lead is not None:")
    assert hold_idx < lead_idx


def test_lead_anchor_position_exempt_when_name_equals_corridor_dest():
    # HRI duality: the guarded hazard's name can equal the corridor
    # destination name (the hand is hazard AND destination). The primary
    # guard position (obstacle_pos) IS extrapolated -- it's the guard entry
    # -- but the corridor anchor slot (dest_pos) must stay at the OBSERVED
    # position, never the predictive anchor, mirroring the Task 1/2 "corridor
    # anchors are never gated" contract. This replicates the wiring's
    # `_lead_pos` vs `_lead_anchor_pos` split (run_guided_libero_safety_eval.py).
    store = {"hand": np.array([0.20, 0.0, 1.0])}
    prev_store = {"hand": np.array([0.19, 0.0, 1.0])}
    guard_pos = _pos_of(store)
    guard = ["hand"]
    corridor_dest = "hand"  # duality: hazard name == destination name
    corridor_target = None
    lead_adj = ls._lead_positions(
        {n: guard_pos(n) for n in guard}, prev_store, {"hand"},
        lead_steps=3.0, cap=LEAD_CAP)

    def lead_pos(name):
        return lead_adj.get(name, guard_pos(name))

    def lead_anchor_pos(name, _exempt=(corridor_target, corridor_dest)):
        if name in _exempt:
            return guard_pos(name)
        return lead_adj.get(name, guard_pos(name))

    g0 = lead_pos(guard[0])
    dest_pos = lead_anchor_pos(corridor_dest)
    np.testing.assert_allclose(g0, [0.23, 0.0, 1.0])  # extrapolated
    np.testing.assert_allclose(dest_pos, store["hand"])  # observed, unchanged
    assert not np.allclose(g0, dest_pos)


def test_defaults_off_mover_lead_leaves_payload_path_untouched():
    # With --mover_lead unset (argparse default None), the wiring never
    # calls _lead_positions; `_lead_pos` stays bound to the raw `_guard_pos`
    # lookup, so g0/obstacle2 positions are exactly the observed positions —
    # byte-identical to Task 2 behavior.
    guard = ["static_obj", "moving_hand"]
    store = {"static_obj": np.array([0.2, 0.0, 1.0]),
             "moving_hand": np.array([-0.2, 0.0, 1.0])}
    eef = np.array([0.0, 0.0, 1.0])
    guard_pos = _pos_of(store)
    mover_lead = None  # the argparse default
    lead_pos = guard_pos
    if mover_lead is not None:
        adj = ls._lead_positions({n: guard_pos(n) for n in guard},
                                  {}, {"moving_hand"}, mover_lead, LEAD_CAP)
        lead_pos = lambda name: adj.get(name, guard_pos(name))  # noqa: E731
    g0 = lead_pos(guard[0]) if guard else None
    np.testing.assert_allclose(g0, store["static_obj"])
    assert lead_pos is guard_pos
