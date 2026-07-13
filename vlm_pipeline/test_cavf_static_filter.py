import numpy as np

from cavf_static_filter import CAVFStaticSafetyFilter, CAVFStaticConfig


def _obs(eef=(0.0, 0.0, 0.9), obstacle=(0.12, 0.0, 0.9)):
    return {
        "robot0_eef_pos": np.array(eef, dtype=float),
        "mug_obstacle_pos": np.array(obstacle, dtype=float),
    }


def _filter(obs):
    safety_filter = CAVFStaticSafetyFilter(
        CAVFStaticConfig(
            translation_scale=0.050,
            safety_margin=0.015,
            influence_distance=0.050,
            smoothing=0.0,
            route_gain=1.2,
        )
    )
    safety_filter.reset(obs, "move the bowl past the obstacle", "mug_obstacle")
    return safety_filter


def test_safe_nominal_chunk_is_preserved():
    obs = _obs(obstacle=(0.2, 0.25, 0.9))
    safety_filter = _filter(obs)
    actions = np.zeros((8, 7), dtype=np.float32)
    actions[:, 0] = 0.1
    filtered, decision = safety_filter.filter_chunk(actions, obs)
    np.testing.assert_allclose(filtered, actions, atol=1e-7)
    assert not decision.intervened
    assert decision.reason == "cavf_nominal_safe"


def test_segment_collision_receives_a_viable_route_repair():
    # A long action can cross an obstacle even when its endpoints are clear.
    # The initial end-effector is outside the conservative safety envelope,
    # but the nominal chunk crosses through it.
    obs = _obs(obstacle=(0.15, 0.0, 0.9))
    safety_filter = _filter(obs)
    actions = np.zeros((5, 7), dtype=np.float64)
    actions[:, 0] = 0.7
    actions[:, 3:6] = 0.1
    actions[:, 6] = -1.0
    filtered, decision = safety_filter.filter_chunk(actions, obs)
    assert decision.intervened
    assert not decision.infeasible
    assert decision.min_clearance_filtered >= -1e-6
    assert decision.min_clearance_filtered > decision.min_clearance_nominal
    assert decision.reason.startswith("cavf_viability_route_")
    np.testing.assert_array_equal(filtered[:, 3:], actions[:, 3:])


def test_candidate_ranker_records_multiple_routes_and_keeps_dtype():
    obs = _obs(obstacle=(0.10, 0.0, 0.9))
    safety_filter = _filter(obs)
    actions = np.zeros((8, 7), dtype=np.float32)
    actions[:, 0] = 0.5
    filtered, _ = safety_filter.filter_chunk(actions, obs)
    assert filtered.dtype == actions.dtype
    diagnostics = safety_filter.metrics.candidate_diagnostics[0]
    assert {item["route"] for item in diagnostics} >= {"nominal", "left", "right", "above"}


def test_invalid_chunk_fails_closed_before_any_projection():
    safety_filter = _filter(_obs())
    with np.testing.assert_raises(ValueError):
        safety_filter.filter_chunk(np.zeros(7), _obs())
