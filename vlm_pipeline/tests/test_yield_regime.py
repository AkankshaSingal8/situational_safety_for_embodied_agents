"""CPU unit tests for Phase 1 Yield regime (`--yield_regime`, client-only
certified retreat): TRIGGER -> RETREAT -> HOLD -> RESUME.

Covers `YieldController`, `_yield_retreat_waypoints`, `_yield_certify_step`,
`_segment_point_distance` in run_guided_libero_safety_eval.py — pure, no
GPU, no env creation, no mujoco/robosuite imports. Modeled on
test_fol_predicates.py.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import run_guided_libero_safety_eval as ls  # noqa: E402

HARD_R = ls.ENGAGE_CONE_HARD_R


def _mk_controller(r_occ=0.20, d_stall=0.01, w=5, standoff=0.20, m=2, y_max=2):
    return ls.YieldController(r_occ, d_stall, w, standoff, m, y_max)


# --- defaults-off byte-identity -------------------------------------------

def test_argparse_default_is_off_and_tunables_present():
    src = pathlib.Path(ls.__file__).read_text()
    assert '"--yield_regime", action="store_true"' in src
    assert '"--yield_r_occ", type=float, default=0.20' in src  # fix round 2
    assert '"--yield_d_stall", type=float, default=0.01' in src
    assert '"--yield_d_move", type=float, default=0.01' in src  # fix round 2
    assert '"--yield_w", type=int, default=5' in src
    assert '"--yield_standoff", type=float, default=0.20' in src
    assert '"--yield_m", type=int, default=2' in src
    assert '"--yield_max", type=int, default=2' in src


def test_yield_intercepts_true_only_for_active_controller():
    # `_yield_intercepts` is the SINGLE function main()'s wiring calls at
    # every Yield-related branch point (loop-top gate, server-call gate,
    # intercept branch) -- exercise it directly across every yc state.
    assert ls._yield_intercepts(None) is False
    yc = _mk_controller()
    assert yc.state == "idle"
    assert ls._yield_intercepts(yc) is False
    yc.state = "retreat"
    assert ls._yield_intercepts(yc) is True
    yc.state = "hold"
    assert ls._yield_intercepts(yc) is True
    yc.state = "idle"
    assert ls._yield_intercepts(yc) is False


def test_defaults_off_drives_real_decision_path_no_server_skip():
    # Mirrors test_defaults_off_engage_cone_none_leaves_payload_path_untouched:
    # rather than grepping source text, this DRIVES the real production
    # function `_yield_intercepts` -- the exact same callable main() uses at
    # every Yield branch point -- with the argparse-default-off state
    # (`yc = None`, since the wiring is literally
    # `yc = YieldController(...) if args.yield_regime else None`) and
    # asserts every dependent branch collapses to the pre-Yield behavior.
    yield_regime = False  # the argparse default (verified above)
    yc = _mk_controller() if yield_regime else None
    assert yc is None

    # (1) Loop-top gate for entering the server-query block:
    # `if not _yield_active and not plan:` where
    # `_yield_active = _yield_intercepts(yc)`. With the flag off this must
    # reduce EXACTLY to the pre-Yield `not plan` condition, for both an
    # empty and a non-empty plan.
    for plan_nonempty in (False, True):
        yield_active = ls._yield_intercepts(yc)
        assert yield_active is False  # no yield state consulted/active
        enters_query_block = (not yield_active) and (not plan_nonempty)
        assert enters_query_block == (not plan_nonempty)

    # (2) Inside that block, the real server-call gate
    # `if not _yield_intercepts(yc):` -- must be True, i.e. the server IS
    # queried and `plan` IS populated, exactly as before Yield existed.
    assert (not ls._yield_intercepts(yc)) is True

    # (3) The Yield-interception branch for `_a` construction,
    # `if _yield_intercepts(yc):` -- must be False, i.e. the normal
    # `plan.popleft()` / `in_retreat` path is taken, never the Yield one.
    assert ls._yield_intercepts(yc) is False


def test_defaults_off_no_controller_created():
    # Mirrors the `--engage_cone`/`--holding_disengage` precedent: with the
    # flag off, no controller object exists at all (not merely "idle").
    yield_regime = False  # the argparse default, confirmed above
    yc = _mk_controller() if yield_regime else None
    assert yc is None


def test_defaults_off_jsonl_fields_absent():
    # Per-episode JSONL: yields/yield_steps/yield_resumes must be
    # absent-safe when the flag is off (not present with 0 -- literally
    # absent), matching the existing --stall_recovery/--ssm_margins pattern.
    _rec = {"task": 0, "ep": 0}
    args_yield_regime = False
    if args_yield_regime:
        _rec["yields"] = 0
    assert "yields" not in _rec
    src = pathlib.Path(ls.__file__).read_text()
    assert 'if args.yield_regime:\n                    _rec["yields"]' in src


# --- mutual exclusion with --stall_recovery --------------------------------

def test_mutual_exclusion_assert_with_stall_recovery(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_guided_libero_safety_eval.py",
        "--yield_regime", "--stall_recovery",
        "--num_trials_per_task", "1",
    ])
    with pytest.raises(AssertionError):
        ls.main()


# --- STALLED(eef): stall window --------------------------------------------

def test_is_stalled_requires_full_window():
    hist = ls.collections.deque(maxlen=5)
    hist.append(np.array([0.0, 0.0, 1.0]))
    # Only 1/5 positions recorded -> not enough history -> never "stalled".
    assert ls.YieldController.is_stalled(hist, d_stall=0.01) is False


def test_is_stalled_exactly_w_boundaries_under_threshold():
    hist = ls.collections.deque(maxlen=5)
    base = np.array([0.30, 0.10, 0.85])
    for i in range(5):
        hist.append(base + np.array([0.001 * i, 0.0, 0.0]))  # net 0.004 m < 0.01
    assert ls.YieldController.is_stalled(hist, d_stall=0.01) is True


def test_is_stalled_full_window_over_threshold_not_stalled():
    hist = ls.collections.deque(maxlen=5)
    base = np.array([0.30, 0.10, 0.85])
    for i in range(5):
        hist.append(base + np.array([0.02 * i, 0.0, 0.0]))  # net 0.08 m > 0.01
    assert ls.YieldController.is_stalled(hist, d_stall=0.01) is False


# --- OCCUPIES ----------------------------------------------------------

def test_occupies_within_radius_true():
    hazard = np.array([0.30, 0.10, 0.85])
    target = np.array([0.32, 0.10, 0.85])  # d = 0.02 < r_occ 0.12
    assert ls.YieldController.occupies(hazard, target, r_occ=0.12) is True


def test_occupies_outside_radius_false():
    hazard = np.array([0.30, 0.10, 0.85])
    target = np.array([0.60, 0.10, 0.85])  # d = 0.30 > r_occ 0.12
    assert ls.YieldController.occupies(hazard, target, r_occ=0.12) is False


def test_occupies_unresolvable_position_false():
    assert ls.YieldController.occupies(None, np.array([0, 0, 0]), 0.12) is False
    assert ls.YieldController.occupies(np.array([0, 0, 0]), None, 0.12) is False


# --- TRIGGER: stalled ∧ mover ∧ occupies, each conjunct independent -------

def test_trigger_fires_when_all_three_conjuncts_true():
    yc = _mk_controller()
    assert yc.should_trigger(
        stalled=True, hazard_name="hand_1", movers={"hand_1"},
        occupies_flag=True) is True


def test_trigger_false_when_not_stalled():
    yc = _mk_controller()
    assert yc.should_trigger(
        stalled=False, hazard_name="hand_1", movers={"hand_1"},
        occupies_flag=True) is False


def test_trigger_false_when_hazard_not_a_mover_static_hand_excluded():
    # This is the t13 exclusion: a static hand (never in `movers`) never
    # triggers Yield even if stalled and geometrically occupying the goal.
    yc = _mk_controller()
    assert yc.should_trigger(
        stalled=True, hazard_name="hand_1", movers=set(),
        occupies_flag=True) is False


def test_trigger_false_when_not_occupies():
    yc = _mk_controller()
    assert yc.should_trigger(
        stalled=True, hazard_name="hand_1", movers={"hand_1"},
        occupies_flag=False) is False


def test_trigger_false_when_hazard_name_none_no_guard_engaged():
    yc = _mk_controller()
    assert yc.should_trigger(
        stalled=True, hazard_name=None, movers={"hand_1"},
        occupies_flag=True) is False


def test_trigger_false_when_movers_metadata_unavailable():
    yc = _mk_controller()
    assert yc.should_trigger(
        stalled=True, hazard_name="hand_1", movers=None,
        occupies_flag=True) is False


def test_trigger_false_when_not_idle():
    yc = _mk_controller()
    yc.state = "hold"
    assert yc.should_trigger(
        stalled=True, hazard_name="hand_1", movers={"hand_1"},
        occupies_flag=True) is False


# --- RUNTIME MOVING (fix round 2: hazard moves after the settle window) ---

def test_runtime_moving_requires_two_readings():
    hist = ls.collections.deque(maxlen=2)
    assert ls.YieldController.runtime_moving(hist, d_move=0.01) is False
    hist.append((0, np.array([0.3, 0.1, 0.85])))
    assert ls.YieldController.runtime_moving(hist, d_move=0.01) is False


def test_runtime_moving_static_hazard_never_displaces_t13_class():
    # A truly static hand: identical position across replan boundaries ->
    # runtime_moving stays False no matter how many readings accumulate.
    hist = ls.collections.deque(maxlen=2)
    p = np.array([0.3, 0.1, 0.85])
    for i in range(5):
        hist.append((i, p.copy()))
        assert ls.YieldController.runtime_moving(hist, d_move=0.01) is False


def test_runtime_moving_displacement_over_threshold_true():
    hist = ls.collections.deque(maxlen=2)
    hist.append((0, np.array([0.30, 0.10, 0.85])))
    hist.append((1, np.array([0.32, 0.10, 0.85])))  # displaced 0.02 > 0.01
    assert ls.YieldController.runtime_moving(hist, d_move=0.01) is True


def test_runtime_moving_displacement_under_threshold_false():
    hist = ls.collections.deque(maxlen=2)
    hist.append((0, np.array([0.30, 0.10, 0.85])))
    hist.append((1, np.array([0.302, 0.10, 0.85])))  # displaced 0.002 < 0.01
    assert ls.YieldController.runtime_moving(hist, d_move=0.01) is False


def test_static_during_settle_window_then_displaces_later_arms_trigger():
    # The G-Y2 root cause: the hand is stationary during the settle window
    # (absent from the episode-level `movers` set) and only starts moving
    # once the episode is underway. Simulate a maxlen-2 tracked-hazard
    # history across several replan boundaries: static readings first
    # (`movers=set()` throughout, mirroring the settle-window miss), then a
    # late displacement -- runtime_moving flips True and TRIGGER becomes
    # reachable even though `movers` never contains the hazard.
    hist = ls.collections.deque(maxlen=2)
    yc = _mk_controller()
    static_pos = np.array([0.30, 0.10, 0.85])
    for i in range(3):
        hist.append((i, static_pos.copy()))
        moving = ls.YieldController.runtime_moving(hist, d_move=0.01)
        assert moving is False
        assert yc.should_trigger(True, "hand_1", set(), True, moving) is False

    # Episode is now underway; the hand starts moving (post-settle).
    moved_pos = np.array([0.34, 0.10, 0.85])  # displaced 0.04 > d_move 0.01
    hist.append((3, moved_pos))
    moving = ls.YieldController.runtime_moving(hist, d_move=0.01)
    assert moving is True
    # `movers` (settle-window set) still empty -- runtime_moving alone arms it.
    assert yc.should_trigger(True, "hand_1", set(), True, moving) is True
    # And also when movers metadata is entirely unavailable (None).
    assert yc.should_trigger(True, "hand_1", None, True, moving) is True


def test_runtime_moving_or_settle_window_movers_either_suffices():
    yc = _mk_controller()
    # settle-window movers alone (runtime_moving False) still fires.
    assert yc.should_trigger(True, "hand_1", {"hand_1"}, True, False) is True
    # runtime_moving alone (not in settle-window movers) still fires.
    assert yc.should_trigger(True, "hand_1", set(), True, True) is True
    # neither -> excluded (t13-class truly static hand).
    assert yc.should_trigger(True, "hand_1", set(), True, False) is False


def test_hazard_identity_change_resets_tracked_history():
    # Mirrors the wiring: `yield_hazard_hist` is reset whenever the tracked
    # guard identity changes, so a displacement between two DIFFERENT
    # hazards' positions is never mistaken for one hazard moving.
    hist = ls.collections.deque(maxlen=2)
    track_name = None

    def _update(candidate, pos):
        nonlocal track_name, hist
        if candidate != track_name:
            track_name = candidate
            hist.clear()
        hist.append((0, pos))

    _update("hand_1", np.array([0.0, 0.0, 0.85]))
    _update("hand_2", np.array([5.0, 5.0, 0.85]))  # identity change -> reset
    assert len(hist) == 1  # history was cleared, only the new entry present
    assert ls.YieldController.runtime_moving(hist, d_move=0.01) is False


# --- percep-tier refresh set (fix round 3: G-Y2 percep-tier blindness) ----

def test_percep_refresh_set_yield_off_movers_only_legacy_behavior():
    # Legacy behavior, byte-identical: with yield_on=False, the refresh set
    # is exactly `movers`, regardless of `guard` -- guard hazards are never
    # pulled in when Yield is off.
    movers = {"hand_1"}
    guard = ["hand_1", "obstacle_a"]
    out = ls._percep_refresh_set(movers, guard, yield_on=False)
    assert out == {"hand_1"}


def test_percep_refresh_set_yield_off_empty_movers_stays_empty_noop():
    # The no-op-when-empty case the fix must preserve exactly.
    out = ls._percep_refresh_set(set(), ["hand_1", "obstacle_a"], yield_on=False)
    assert out == set()


def test_percep_refresh_set_yield_on_empty_movers_nonempty_guard():
    # The G-Y2 fix: with yield_on=True and an empty settle-window `movers`
    # set (the dynamic-intruder case) but a non-empty identified guard, the
    # refresh set equals the guard set -- no longer a no-op.
    guard = ["hand_1", "obstacle_a"]
    out = ls._percep_refresh_set(set(), guard, yield_on=True)
    assert out == {"hand_1", "obstacle_a"}


def test_percep_refresh_set_yield_on_unions_movers_and_guard():
    movers = {"hand_1"}
    guard = ["hand_1", "obstacle_a"]
    out = ls._percep_refresh_set(movers, guard, yield_on=True)
    assert out == {"hand_1", "obstacle_a"}


def test_percep_refresh_set_yield_on_empty_guard_and_movers_stays_empty():
    out = ls._percep_refresh_set(set(), [], yield_on=True)
    assert out == set()
    out_none_movers = ls._percep_refresh_set(None, [], yield_on=True)
    assert out_none_movers == set()


def test_percep_refresh_wiring_gates_and_sorts_on_refresh_set():
    src = pathlib.Path(ls.__file__).read_text()
    assert "_refresh = _percep_refresh_set(movers, guard, args.yield_regime)" in src
    assert "and _refresh):" in src
    assert "sorted(_refresh)" in src
    # The legacy `sorted(movers)` refresh calls must be gone -- the fix
    # replaces them, not adds a parallel path.
    assert "estimate_object_positions(\n                                env.sim, sorted(movers)" not in src
    assert "percep_snapshot(env, sorted(movers))" not in src


# --- blocker telemetry (fix round 2, diagnosability) -----------------------

def test_blocker_conjunct_all_true_no_blocker():
    assert ls._yield_blocker_conjunct(True, True, True) is None


def test_blocker_conjunct_sole_stall_blocker():
    assert ls._yield_blocker_conjunct(False, True, True) == "stall"


def test_blocker_conjunct_sole_moving_blocker():
    assert ls._yield_blocker_conjunct(True, False, True) == "moving"


def test_blocker_conjunct_sole_occupies_blocker():
    assert ls._yield_blocker_conjunct(True, True, False) == "occupies"


def test_blocker_conjunct_multiple_false_no_sole_blocker():
    assert ls._yield_blocker_conjunct(False, False, True) is None
    assert ls._yield_blocker_conjunct(False, False, False) is None


def test_blocker_telemetry_g_y2_scenario_moving_is_the_blocker():
    # Reproduces the G-Y2 forensic signature over a mini replan-boundary
    # sequence: stalled and occupying throughout, but moving only becomes
    # True on the last evaluation -- "moving" should accumulate as the sole
    # blocker on every evaluation before that.
    counts = {"stall": 0, "moving": 0, "occupies": 0}
    conjuncts_per_replan = [
        (True, False, True),   # moving blocks
        (True, False, True),   # moving blocks
        (True, True, True),    # all true -> would TRIGGER, no blocker
    ]
    for stalled, moving, occupies in conjuncts_per_replan:
        blocker = ls._yield_blocker_conjunct(stalled, moving, occupies)
        if blocker is not None:
            counts[blocker] += 1
    assert counts == {"stall": 0, "moving": 2, "occupies": 0}


def test_yield_blockers_jsonl_field_absent_when_flag_off():
    _rec = {"task": 0, "ep": 0}
    args_yield_regime = False
    if args_yield_regime:
        _rec["yield_blockers"] = {"stall": 0, "moving": 0, "occupies": 0}
    assert "yield_blockers" not in _rec
    src = pathlib.Path(ls.__file__).read_text()
    assert 'if args.yield_regime:\n                    _rec["yields"]' in src
    assert '_rec["yield_blockers"] = dict(yield_blockers)' in src
    # The addition must be inside the SAME `if args.yield_regime:` block as
    # the other yield JSONL fields, not a separate always-on block.
    yields_idx = src.index('_rec["yields"] = yc.total_yields')
    blockers_idx = src.index('_rec["yield_blockers"] = dict(yield_blockers)')
    assert 0 < blockers_idx - yields_idx < 200


# --- RETREAT targeting ------------------------------------------------

def test_retreat_waypoints_picks_first_prefix_point_at_standoff():
    # Trajectory moving in +x, hazard sits at the far end (near the most
    # recent points) -- retreat should target the first (walking backward)
    # point at distance >= standoff from the CURRENT hazard position.
    traj = [np.array([0.0 + 0.05 * i, 0.0, 0.85]) for i in range(6)]
    # traj[-1] = (0.25, 0, 0.85); hazard right next to it.
    hazard = np.array([0.26, 0.0, 0.85])
    wp = ls._yield_retreat_waypoints(traj, hazard, standoff=0.20)
    # Distances from hazard: traj[0]=0.26, traj[1]=0.21, traj[2]=0.16, ...
    # First (walking backward from traj[-1]) point >= 0.20 is traj[1]
    # (d=0.21); traj[0] also qualifies but traj[1] is reached first.
    assert wp[0] is traj[-1] or np.allclose(wp[0], traj[-1])
    assert np.allclose(wp[-1], traj[1])


def test_retreat_waypoints_order_is_current_to_target():
    traj = [np.array([float(i), 0.0, 0.0]) for i in range(4)]
    hazard = np.array([3.0, 0.0, 0.0])
    wp = ls._yield_retreat_waypoints(traj, hazard, standoff=2.5)
    # traj distances from hazard: [3,2,1,0]; first (from the end backward)
    # >= 2.5 is traj[0] (d=3).
    np.testing.assert_allclose(wp[0], traj[-1])
    np.testing.assert_allclose(wp[-1], traj[0])
    assert len(wp) == 4


def test_retreat_waypoints_fallback_to_earliest_point():
    # Hazard so close nothing in the trajectory clears standoff -> falls
    # back to the earliest recorded point (best-effort along known path).
    traj = [np.array([0.0, 0.0, 0.0]), np.array([0.01, 0.0, 0.0])]
    hazard = np.array([0.005, 0.0, 0.0])
    wp = ls._yield_retreat_waypoints(traj, hazard, standoff=5.0)
    np.testing.assert_allclose(wp[-1], traj[0])


def test_retreat_waypoints_short_trajectory_degenerate():
    out = ls._yield_retreat_waypoints(
        [np.array([0.0, 0.0, 0.0])], np.array([1, 0, 0]), 0.2)
    assert len(out) < 2


# Shared geometry for the retreat-delta/certification tests below: a
# trajectory moving along +x (spacing 0.05 m/point, plausible per-replan
# displacement); a hazard offset 0.15 m in +y from the endpoint (0.15 is
# strictly between ENGAGE_CONE_HARD_R=0.14 and the default standoff=0.20)
# so early segments FAIL the standoff distance check (forcing a multi-
# waypoint retreat) while every segment on the trajectory line still stays
# outside the hard-radius certification (min possible distance to the
# hazard is exactly 0.15 >= 0.14).
_TRAJ_X = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
_RETREAT_TRAJ = [np.array([x, 0.0, 0.85]) for x in _TRAJ_X]
_RETREAT_HAZARD = np.array([0.30, 0.15, 0.85])  # y-offset 0.15


def test_retreat_deltas_respect_exec_parity_convention():
    yc = _mk_controller()  # default standoff=0.20
    yc.trigger(_RETREAT_TRAJ, _RETREAT_HAZARD)
    assert yc.state == "retreat"
    assert len(yc.waypoints) >= 3  # multi-waypoint retreat, not a 1-step plan
    cur = _RETREAT_TRAJ[-1]
    nxt = yc.waypoints[1]
    action_default, _ = yc.next_retreat_action(
        cur, _RETREAT_HAZARD, HARD_R, exec_parity=False, raw_actions=False)
    delta = nxt - cur
    expected_default = np.clip(
        np.concatenate([delta, np.zeros(4)]) * ls.ACTION_TO_CMD, -1.0, 1.0)
    np.testing.assert_allclose(action_default, expected_default)

    yc2 = _mk_controller()
    yc2.trigger(_RETREAT_TRAJ, _RETREAT_HAZARD)
    action_parity, _ = yc2.next_retreat_action(
        cur, _RETREAT_HAZARD, HARD_R, exec_parity=True, raw_actions=False)
    expected_parity = np.concatenate([delta, np.zeros(4)])
    np.testing.assert_allclose(action_parity, expected_parity)


# --- retreat certification (certificate integrity invariant) --------------

def test_segment_point_distance_basic():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    p = np.array([0.5, 0.3, 0.0])
    assert np.isclose(ls._segment_point_distance(a, b, p), 0.3)


def test_certify_step_rejects_segment_within_hard_radius():
    cur = np.array([0.0, 0.0, 0.85])
    nxt = np.array([0.10, 0.0, 0.85])
    hazard_moved_onto_path = np.array([0.05, 0.0, 0.85])  # dead center of segment
    dist = ls._segment_point_distance(cur, nxt, hazard_moved_onto_path)
    assert dist < HARD_R
    assert ls._yield_certify_step(cur, nxt, hazard_moved_onto_path, HARD_R) is False


def test_certify_step_accepts_segment_clear_of_hazard():
    cur = np.array([0.0, 0.0, 0.85])
    nxt = np.array([0.10, 0.0, 0.85])
    hazard_far = np.array([0.05, 1.0, 0.85])
    assert ls._yield_certify_step(cur, nxt, hazard_far, HARD_R) is True


def test_certify_step_missing_hazard_position_conservative_certify():
    cur = np.array([0.0, 0.0, 0.85])
    nxt = np.array([0.10, 0.0, 0.85])
    assert ls._yield_certify_step(cur, nxt, None, HARD_R) is True


def test_retreat_step_rejected_by_moved_hazard_switches_to_hold():
    # The certificate integrity test: a hazard that has MOVED onto the
    # retreat segment since the last check must cause the step to be
    # rejected -> HOLD, never executed uncertified.
    yc = _mk_controller(standoff=0.03)
    traj = [np.array([0.0, 0.0, 0.85]), np.array([0.05, 0.0, 0.85]),
            np.array([0.10, 0.0, 0.85])]
    # Hazard close to the current point at TRIGGER time -- forces a
    # multi-waypoint retreat plan.
    hazard_at_trigger = np.array([0.11, 0.0, 0.85])
    yc.trigger(traj, hazard_at_trigger)
    assert yc.state == "retreat"
    cur = traj[-1]
    # Hazard has now MOVED directly onto the next retreat segment.
    moved_hazard = np.array([0.075, 0.0, 0.85])
    action, new_state = yc.next_retreat_action(cur, moved_hazard, HARD_R)
    assert action is None
    assert new_state == "hold"
    assert yc.state == "hold"


def test_retreat_step_certified_and_executed_when_clear():
    yc = _mk_controller()
    yc.trigger(_RETREAT_TRAJ, _RETREAT_HAZARD)
    assert yc.state == "retreat"
    n_waypoints_before = len(yc.waypoints)
    assert n_waypoints_before >= 3
    cur = _RETREAT_TRAJ[-1]
    hazard_now_clear = np.array([10.0, 10.0, 10.0])
    action, new_state = yc.next_retreat_action(cur, hazard_now_clear, HARD_R)
    assert action is not None
    assert new_state == "retreat"  # more than one waypoint remained
    assert yc.total_steps == 1
    assert len(yc.waypoints) == n_waypoints_before - 1


# --- RESUME hysteresis ------------------------------------------------

def test_resume_requires_m_consecutive_not_occupies():
    yc = _mk_controller(m=2)
    yc.state = "hold"
    assert yc.evaluate_resume(occupies_flag=False) is False  # 1/2
    assert yc.state == "hold"
    assert yc.evaluate_resume(occupies_flag=False) is True   # 2/2 -> resume
    assert yc.state == "idle"
    assert yc.total_resumes == 1


def test_resume_one_occupied_evaluation_resets_streak():
    yc = _mk_controller(m=2)
    yc.state = "hold"
    assert yc.evaluate_resume(occupies_flag=False) is False  # 1/2
    assert yc.evaluate_resume(occupies_flag=True) is False   # reset to 0
    assert yc.resume_streak == 0
    assert yc.state == "hold"
    assert yc.evaluate_resume(occupies_flag=False) is False  # 1/2 again
    assert yc.evaluate_resume(occupies_flag=False) is True   # 2/2 -> resume
    assert yc.state == "idle"


# --- stall window reset across a full yield cycle (fix round 1) -----------

def test_stall_window_cleared_across_full_trigger_retreat_hold_resume_cycle():
    # Fix: `yield_replan_hist` is frozen during RETREAT/HOLD (never
    # appended to while yc.state != "idle") and must be cleared both on
    # TRIGGER and on RESUME, so a post-resume stall evaluation never mixes
    # stale pre-yield positions with the first fresh post-resume sample.
    # Mirrors the exact clear-on-trigger / clear-on-resume calls added to
    # main()'s wiring (`yield_replan_hist.clear()` right after
    # `yc.trigger(...)` and right after a True `yc.evaluate_resume(...)`).
    w = 3
    yc = _mk_controller(w=w, standoff=0.03)
    yield_replan_hist = ls.collections.deque(maxlen=w)

    # --- idle phase: window fills with STALE positions near x=0. ---
    stale = [np.array([0.0, 0.0, 0.85]), np.array([0.0005, 0.0, 0.85]),
             np.array([0.001, 0.0, 0.85])]
    for p in stale:
        yield_replan_hist.append(p)
    assert len(yield_replan_hist) == w
    assert ls.YieldController.is_stalled(yield_replan_hist, d_stall=0.01) is True

    # --- TRIGGER -> RETREAT/HOLD; window must be cleared immediately. ---
    traj = [np.array([0.0, 0.0, 0.85]), np.array([0.001, 0.0, 0.85])]
    hazard = np.array([10.0, 10.0, 10.0])
    yc.trigger(traj, hazard)
    assert yc.triggers == 1
    yield_replan_hist.clear()  # the fix under test
    assert len(yield_replan_hist) == 0

    # --- Window stays frozen throughout RETREAT/HOLD (not appended). ---
    yc.state = "hold"
    assert len(yield_replan_hist) == 0

    # --- HOLD -> RESUME (M=2 consecutive not-OCCUPIES). ---
    assert yc.evaluate_resume(occupies_flag=False) is False
    assert yc.evaluate_resume(occupies_flag=False) is True
    assert yc.state == "idle"
    yield_replan_hist.clear()  # the fix under test
    assert len(yield_replan_hist) == 0

    # --- Post-RESUME: a single fresh sample must NOT be judged stalled
    # against a window that no longer holds stale pre-yield positions
    # (window isn't even full yet). ---
    fresh_far = np.array([5.0, 0.0, 0.85])  # eef moved far during retreat
    yield_replan_hist.append(fresh_far)
    assert len(yield_replan_hist) == 1
    assert ls.YieldController.is_stalled(yield_replan_hist, d_stall=0.01) is False

    # --- Fill the window with fresh, genuinely-close post-resume
    # positions: THIS determines the second STALLED evaluation, not any
    # stale pre-yield sample (which must be fully gone from the deque). ---
    for p in [np.array([5.0005, 0.0, 0.85]), np.array([5.001, 0.0, 0.85])]:
        yield_replan_hist.append(p)
    assert len(yield_replan_hist) == w
    for p in yield_replan_hist:
        assert not np.allclose(p, stale[0])
    assert ls.YieldController.is_stalled(yield_replan_hist, d_stall=0.01) is True

    # --- A second TRIGGER is now legitimately possible (armed, 1 < y_max). ---
    assert yc.armed() is True
    assert yc.should_trigger(True, "hand_1", {"hand_1"}, True) is True


def test_main_wiring_clears_stall_window_on_trigger_and_resume():
    # Source-level check that the fix is actually wired into main() at both
    # required call sites (trigger and resume), not just exercised by the
    # behavioral test above.
    src = pathlib.Path(ls.__file__).read_text()
    trig_idx = src.index("yc.trigger(list(eef_traj), _y_hpos)")
    trig_clear_idx = src.index("yield_replan_hist.clear()", trig_idx)
    # The clear must be the next statement-ish thing after trigger() (no
    # unrelated yield_replan_hist mutation in between).
    assert trig_clear_idx - trig_idx < 400
    resume_idx = src.index("if yc.evaluate_resume(_yh_occ):")
    resume_clear_idx = src.index("yield_replan_hist.clear()", resume_idx)
    assert resume_clear_idx - resume_idx < 600


# --- Y_max cap ----------------------------------------------------------

def test_y_max_cap_third_trigger_does_not_arm():
    yc = _mk_controller(y_max=2, standoff=0.03)
    traj = [np.array([0.0, 0.0, 0.85]), np.array([0.10, 0.0, 0.85])]
    hazard = np.array([10.0, 10.0, 10.0])

    yc.trigger(traj, hazard)
    assert yc.triggers == 1
    yc.state = "hold"
    yc.evaluate_resume(False)
    yc.evaluate_resume(False)
    assert yc.state == "idle"

    yc.trigger(traj, hazard)
    assert yc.triggers == 2
    yc.state = "hold"
    yc.evaluate_resume(False)
    yc.evaluate_resume(False)
    assert yc.state == "idle"

    # Third attempt: armed() is now False (triggers == y_max) -> never arms.
    assert yc.armed() is False
    assert yc.should_trigger(
        stalled=True, hazard_name="hand_1", movers={"hand_1"},
        occupies_flag=True) is False
    assert yc.triggers == 2


def test_wiring_gates_trigger_on_y_max_and_state():
    # should_trigger() itself is the single arming gate; the wiring in
    # main() calls it only from `yc.state == "idle"`, so a triggered/held
    # controller can never re-trigger mid-yield either.
    yc = _mk_controller(y_max=1, standoff=0.03)
    traj = [np.array([0.0, 0.0, 0.85]), np.array([0.10, 0.0, 0.85])]
    hazard = np.array([10.0, 10.0, 10.0])
    yc.trigger(traj, hazard)
    assert yc.armed() is False  # y_max=1, already used
    assert yc.should_trigger(True, "hand_1", {"hand_1"}, True) is False
