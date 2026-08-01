"""CPU unit tests for the LS runner's corridor-anchor / top-k guard helpers.

No GPU, no env creation, no mujoco/robosuite imports: the helpers under test
(`_corridor_anchor`, `_topk_guard`, `_extend_guidance`) are pure functions in
run_guided_libero_safety_eval.py (heavy imports live inside main()).
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import run_guided_libero_safety_eval as ls  # noqa: E402

EEF = np.array([0.0, 0.0, 1.0])

# Stubbed episode-start candidates (obs `{name}_pos` after workspace filter).
CANDS = {
    "banana_1": np.array([0.10, 0.10, 0.90]),
    "plate_1": np.array([0.20, 0.30, 0.90]),
    "moka_pot_obstacle": np.array([0.05, 0.08, 0.92]),
    "ball_obstacle": np.array([0.30, -0.20, 0.90]),
}


def test_corridor_anchor_picks_task_referenced_object():
    target, dest = ls._corridor_anchor("put the banana on the plate", CANDS)
    assert target == "banana_1"          # sanctioned goal object = anchor
    assert dest == "plate_1"             # latest head-noun = destination


def test_corridor_anchor_missing_entities_fail_safe():
    target, dest = ls._corridor_anchor(
        "press the button", {"moka_pot_obstacle": CANDS["moka_pot_obstacle"]})
    assert target is None                # no corridor granted


def test_gt_source_returns_all_checkrobotcontact_args():
    hazards = ["moka_pot_obstacle", "ball_obstacle"]
    guard = ls._topk_guard("gt", "put the banana on the plate",
                           CANDS, EEF, set(), hazards, k=1)
    assert guard == hazards              # today's behavior: ALL hazards, any k


def test_fol_topk2_two_guards_ordered_by_identity_score():
    from symbolic_identity import fol_obstacle_id
    desc = "put the banana on the plate"
    ranked = fol_obstacle_id(desc, CANDS, EEF, moving=set())
    g1 = ls._topk_guard("fol", desc, CANDS, EEF, set(), [], k=1)
    g2 = ls._topk_guard("fol", desc, CANDS, EEF, set(), [], k=2)
    assert len(g2) == 2
    assert g2 == ranked[:2]              # ordered by the identity score
    assert g1 == g2[:1]                  # k=1 unchanged (today's truncation)
    assert set(g2) == {"moka_pot_obstacle", "ball_obstacle"}


def test_symbolic_topk2_primary_unchanged_runnerup_appended():
    from symbolic_identity import scored_obstacle_id
    desc = "put the banana on the plate"
    picked = scored_obstacle_id(desc, CANDS, EEF, moving=set())
    g1 = ls._topk_guard("symbolic", desc, CANDS, EEF, set(), [], k=1)
    g2 = ls._topk_guard("symbolic", desc, CANDS, EEF, set(), [], k=2)
    assert g1 == [picked]                # k=1 identical to today
    assert g2[0] == picked               # primary never changes
    assert len(g2) == 2 and g2[1] != picked


def _base_payload():
    return {
        "enabled": 1.0,
        "eef_pos": np.asarray(EEF, dtype=np.float32),
        "obstacle_pos": np.asarray(CANDS["moka_pot_obstacle"], dtype=np.float32),
        "obstacle_radius": ls.obstacle_radius("moka_pot_obstacle"),
    }


def test_default_payload_shape_single_guard():
    # Defaults (guard_topk=1, corridor anchors as today): exactly today's keys
    # in today's insertion order — no obstacle2.
    out = ls._extend_guidance(_base_payload(), ["moka_pot_obstacle"],
                              CANDS.get, "banana_1", "plate_1")
    assert list(out) == ["enabled", "eef_pos", "obstacle_pos",
                         "obstacle_radius", "target_pos", "dest_pos"]
    assert np.allclose(out["target_pos"], CANDS["banana_1"])
    assert np.allclose(out["dest_pos"], CANDS["plate_1"])


def test_payload_two_guards_emits_obstacle2():
    out = ls._extend_guidance(_base_payload(),
                              ["moka_pot_obstacle", "ball_obstacle"],
                              CANDS.get, None, None)
    assert list(out) == ["enabled", "eef_pos", "obstacle_pos",
                         "obstacle_radius", "obstacle2"]
    assert set(out["obstacle2"]) == {"pos", "radius"}
    assert np.allclose(out["obstacle2"]["pos"], CANDS["ball_obstacle"])
    assert out["obstacle2"]["radius"] == ls.obstacle_radius("ball_obstacle")


def test_payload_missing_positions_fail_safe():
    # Unlocatable second guard / anchors simply add no keys.
    out = ls._extend_guidance(_base_payload(),
                              ["moka_pot_obstacle", "ghost"], CANDS.get,
                              "ghost_target", None)
    assert list(out) == ["enabled", "eef_pos", "obstacle_pos",
                         "obstacle_radius"]


# --- engagement gating (--engage_radius) -----------------------------------
# eef->moka_pot dist ~= 0.124 m, eef->ball dist ~= 0.374 m (from CANDS/EEF).

def _query_payload(guard, radius, target=None, dest=None):
    """Mirror of the episode loop's per-query payload construction:
    gate -> primary position -> base payload -> _extend_guidance.
    Returns None when the loop would send no guidance block."""
    step_guard = ls._gate_guard(guard, CANDS.get, EEF, radius)
    g0 = CANDS.get(step_guard[0]) if step_guard else None
    if g0 is None:
        return None
    payload = {
        "enabled": 1.0,
        "eef_pos": np.asarray(EEF, dtype=np.float32),
        "obstacle_pos": np.asarray(g0, dtype=np.float32),
        "obstacle_radius": ls.obstacle_radius(step_guard[0]),
    }
    return ls._extend_guidance(payload, step_guard, CANDS.get, target, dest)


def _payloads_equal(a, b):
    if a is None or b is None:
        return a is b
    if list(a) != list(b):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, dict):
            if not _payloads_equal(va, vb):
                return False
        elif isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            if not np.array_equal(np.asarray(va), np.asarray(vb)):
                return False
        elif va != vb:
            return False
    return True


def test_engage_radius_zero_is_noop():
    # Default R=0: gate returns the guard unchanged (fresh list) and the
    # payload is identical to today's, for every guard shape.
    for guard in ([], ["moka_pot_obstacle"],
                  ["moka_pot_obstacle", "ball_obstacle"], ["ghost"]):
        gated = ls._gate_guard(guard, CANDS.get, EEF, 0.0)
        assert gated == guard and gated is not guard
        assert _payloads_equal(
            _query_payload(guard, 0.0, "banana_1", "plate_1"),
            _query_payload(guard, -1.0, "banana_1", "plate_1"))


def test_guard_included_when_within_radius():
    out = _query_payload(["moka_pot_obstacle"], 0.15)
    assert out is not None
    assert np.allclose(out["obstacle_pos"], CANDS["moka_pot_obstacle"])


def test_guard_dropped_when_outside_radius():
    # 0.10 < 0.124 m eef->moka distance: no engaged guard -> the query
    # carries no guidance block (same as the --disable_guidance path).
    assert ls._gate_guard(["moka_pot_obstacle"], CANDS.get, EEF, 0.10) == []
    assert _query_payload(["moka_pot_obstacle"], 0.10) is None


def test_obstacle2_gated_independently():
    guard = ["moka_pot_obstacle", "ball_obstacle"]
    # R=0.15: primary engaged (0.124), runner-up out of range (0.374) ->
    # obstacle2 dropped while the primary barrier stays.
    out = _query_payload(guard, 0.15)
    assert list(out) == ["enabled", "eef_pos", "obstacle_pos",
                         "obstacle_radius"]
    assert np.allclose(out["obstacle_pos"], CANDS["moka_pot_obstacle"])
    # R=0.40: both engaged -> identical to the ungated two-guard payload.
    assert _payloads_equal(_query_payload(guard, 0.40),
                           _query_payload(guard, 0.0))
    # Primary out of range, runner-up in range (order swapped): the engaged
    # entry is promoted to the primary slot.
    out = _query_payload(["ball_obstacle", "moka_pot_obstacle"], 0.15)
    assert np.allclose(out["obstacle_pos"], CANDS["moka_pot_obstacle"])
    assert "obstacle2" not in out


def test_corridor_anchors_unaffected_by_gating():
    # Anchors are never distance-gated: present whenever a guard engages...
    out = _query_payload(["moka_pot_obstacle"], 0.15, "banana_1", "plate_1")
    assert np.allclose(out["target_pos"], CANDS["banana_1"])
    assert np.allclose(out["dest_pos"], CANDS["plate_1"])
    # ...even though both anchors lie far outside R (banana 0.17, plate 0.37).
    assert np.linalg.norm(CANDS["banana_1"] - EEF) > 0.15
    # And _gate_guard itself only ever sees/filters guard entries.
    assert ls._gate_guard(["moka_pot_obstacle"], CANDS.get, EEF, 0.15) == \
        ["moka_pot_obstacle"]
