import numpy as np

from contextual_predictive_filter import ContextualPredictiveSafetyFilter, PredictiveFilterConfig


def _obs(eef=(0.0, 0.0, 0.9), obstacle=(0.20, 0.0, 0.9)):
    return {
        "robot0_eef_pos": np.array(eef, dtype=float),
        "mug_obstacle_pos": np.array(obstacle, dtype=float),
    }


def test_safe_chunk_is_unchanged():
    obs = _obs(obstacle=(0.2, 0.3, 0.9))
    filt = ContextualPredictiveSafetyFilter()
    filt.reset(obs, "move bowl", "mug_obstacle")
    actions = np.zeros((8, 7), dtype=np.float32)
    actions[:, 0] = 0.01
    filtered, decision = filt.filter_chunk(actions, obs)
    np.testing.assert_allclose(filtered, actions, atol=1e-7)
    assert not decision.intervened


def test_future_collision_is_redirected_and_gripper_preserved():
    obs = _obs(obstacle=(0.16, 0.0, 0.9))
    filt = ContextualPredictiveSafetyFilter(
        PredictiveFilterConfig(safety_margin=0.02, smoothing=0.0)
    )
    filt.reset(obs, "move bowl", "mug_obstacle")
    actions = np.zeros((8, 7), dtype=np.float64)
    actions[:, 0] = 0.025
    actions[:, 3:6] = 0.1
    actions[:, 6] = -1.0
    filtered, decision = filt.filter_chunk(actions, obs)
    assert decision.intervened
    assert decision.min_clearance_filtered > decision.min_clearance_nominal
    np.testing.assert_array_equal(filtered[:, 3:], actions[:, 3:])


def test_filter_is_action_chunk_model_agnostic():
    obs = _obs()
    filt = ContextualPredictiveSafetyFilter()
    filt.reset(obs, "instruction", "mug_obstacle")
    for horizon in (5, 8, 16):
        actions = np.zeros((horizon, 7), dtype=np.float32)
        filtered, _ = filt.filter_chunk(actions, obs)
        assert filtered.shape == actions.shape
        assert filtered.dtype == actions.dtype


def test_invalid_chunk_fails_closed():
    obs = _obs()
    filt = ContextualPredictiveSafetyFilter()
    filt.reset(obs, "instruction", "mug_obstacle")
    with np.testing.assert_raises(ValueError):
        filt.filter_chunk(np.zeros(7), obs)


def test_valid_low_eef_pose_does_not_trigger_workspace_intervention():
    obs = _obs(eef=(0.0, 0.0, 0.75), obstacle=(0.3, 0.3, 0.9))
    filt = ContextualPredictiveSafetyFilter()
    filt.reset(obs, "instruction", "mug_obstacle")
    actions = np.zeros((8, 7), dtype=np.float32)
    filtered, decision = filt.filter_chunk(actions, obs)
    np.testing.assert_array_equal(filtered, actions)
    assert not decision.intervened
