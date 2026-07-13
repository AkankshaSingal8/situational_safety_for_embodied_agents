from types import SimpleNamespace

import numpy as np

from run_guided_safelibero_pi05_eval import (
    _candidate_score,
    _infer_diverse_candidates,
    _inject_topology_experts,
    _resolve_semantic_context,
    _segment_distance_2d,
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


def test_segment_distance_detects_blocked_contextual_corridor():
    assert _segment_distance_2d([0.5, 0.1], [0.0, 0.0], [1.0, 0.0]) == 0.1
    assert _segment_distance_2d([-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]) == 1.0


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


def test_diverse_candidates_include_nominal_and_geometry_modes():
    class Client:
        def __init__(self):
            self.requests = []

        def infer(self, request):
            self.requests.append(request)
            return {"actions": np.zeros((5, 7)), "guidance": request["guidance"]}

    client = Client()
    element = {"guidance": {"enabled": 1.0, "companion_scale": 1.0, "obstacle_radius": 0.1}}
    _infer_diverse_candidates(client, element, 5)
    guidance = [request["guidance"] for request in client.requests]
    assert guidance[0]["enabled"] == 0.0
    assert guidance[1]["companion_scale"] == 0.0
    assert guidance[2]["companion_scale"] == 1.0
    assert guidance[3]["obstacle_radius"] < 0.1
    assert guidance[4]["obstacle_radius"] > 0.1


def test_transport_phase_injects_opposite_side_tangent_experts():
    obs = _obs()
    initial = obs["akita_black_bowl_2_pos"].copy()
    obs["akita_black_bowl_2_pos"] = initial + np.array([0.0, 0.0, 0.04])
    obs["moka_obstacle_1_pos"] = np.array([0.08, 0.20, 1.0])
    candidates = [{"actions": np.zeros((10, 7))} for _ in range(5)]
    result = _inject_topology_experts(
        candidates, obs, "moka_obstacle_1", "akita_black_bowl_2", initial,
        np.array([0.20, 0.20, 0.90]),
    )
    experts = {candidate.get("expert"): candidate for candidate in result}
    left = experts["tangent_left"]
    right = experts["tangent_right"]
    retreat = experts["radial_retreat"]
    assert np.linalg.norm(left["actions"][0, :2]) > 0.7
    assert np.linalg.norm(right["actions"][0, :2]) > 0.7
    assert np.sign(left["actions"][0, 1]) != np.sign(right["actions"][0, 1])
    assert np.all(left["actions"][:5, 2] == 0.2)
    assert np.linalg.norm(retreat["actions"][0, :2]) > 0.8
    assert np.all(retreat["actions"][:5, 2] >= 0.6)
    assert all(experts[name]["topology_required"] for name in experts if name)
