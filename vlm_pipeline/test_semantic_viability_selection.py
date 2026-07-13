from types import SimpleNamespace

import numpy as np

from run_guided_safelibero_pi05_eval import (
    _candidate_score,
    _resolve_semantic_context,
    _semantic_phase,
)


def _obs():
    return {
        "robot0_eef_pos": np.array([0.0, 0.0, 1.0]),
        "akita_black_bowl_1_pos": np.array([-0.20, 0.20, 0.90]),
        "akita_black_bowl_2_pos": np.array([0.01, 0.20, 0.90]),
        "plate_1_pos": np.array([0.10, 0.20, 0.90]),
        "ramekin_1_pos": np.array([-0.10, 0.20, 0.90]),
        "stove_1_pos": np.array([0.01, 0.20, 0.89]),
    }


def test_relation_resolves_active_bowl_and_goal():
    name, goal = _resolve_semantic_context(
        _obs(), "pick up the black bowl on the stove and place it on the plate"
    )
    assert name == "akita_black_bowl_2"
    np.testing.assert_allclose(goal, [0.10, 0.20, 0.90])


def test_motion_switches_to_transport_phase():
    obs = _obs()
    initial = obs["akita_black_bowl_2_pos"].copy()
    assert _semantic_phase(obs, "akita_black_bowl_2", initial) == "approach"
    obs["akita_black_bowl_2_pos"] = initial + np.array([0.0, 0.0, 0.04])
    assert _semantic_phase(obs, "akita_black_bowl_2", initial) == "transport"


def test_candidate_score_rewards_safe_task_progress():
    args = SimpleNamespace(
        controller_scale=0.01,
        clearance_weight=1.0,
        progress_weight=4.0,
        correction_penalty=0.02,
    )
    toward = {
        "actions": np.array([[1.0, 0.0, 0.0]] * 5),
        "guidance": {"min_clearance": 0.02, "correction_norm": 0.1},
    }
    away = {
        "actions": np.array([[-1.0, 0.0, 0.0]] * 5),
        "guidance": {"min_clearance": 0.02, "correction_norm": 0.1},
    }
    eef = np.array([0.0, 0.0, 0.0])
    target = np.array([0.2, 0.0, 0.0])
    assert _candidate_score(toward, eef, target, args) > _candidate_score(away, eef, target, args)
