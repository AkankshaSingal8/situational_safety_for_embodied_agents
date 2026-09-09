"""S2 / S3: per-episode grounding state must not leak across episodes.

S2 `vlm_grounder.py:192` -- the cache key was task text + sorted object names,
omitting object POSITIONS, while the actual obstacle choice is made by
`symbolic_obstacle_id` from candidate *geometry*. The filter is constructed once
per task and `setup_episode` never cleared the cache, so with SafeLIBERO
randomising object poses across its 50 episodes, episodes 01-49 reused episode
00's obstacle pick.

S3 `filter.py:173` -- `_vision_target_xy` / `_vision_targets_xy` were not reset
in `setup_episode`, so a stale target from a previous episode could relax the
barrier (cancel_scale 0.6) against a phantom target.
"""
import numpy as np

from fol_safety_filter.vlm_grounder import VLMObstacleGrounder
from fol_safety_filter.filter import FOLSafetyFilter


# --- S2: cache key must distinguish layouts ------------------------------

def test_cache_key_distinguishes_different_layouts():
    g = VLMObstacleGrounder(use_vlm=False)
    names = ["moka_pot_obstacle_1", "akita_black_bowl_1"]
    k1 = g._cache_key("pick up the bowl", names,
                      {"moka_pot_obstacle_1": np.array([0.10, 0.20, 0.90])})
    k2 = g._cache_key("pick up the bowl", names,
                      {"moka_pot_obstacle_1": np.array([0.40, 0.20, 0.90])})
    assert k1 != k2, "different object layouts must not share a cache entry"


def test_cache_key_stable_for_identical_layout():
    g = VLMObstacleGrounder(use_vlm=False)
    names = ["moka_pot_obstacle_1"]
    pos = {"moka_pot_obstacle_1": np.array([0.10, 0.20, 0.90])}
    assert g._cache_key("t", names, pos) == g._cache_key("t", names, dict(pos))


def test_cache_key_tolerates_float_noise():
    """Quantised at 1 cm: below SafeLIBERO's randomisation, above float noise."""
    g = VLMObstacleGrounder(use_vlm=False)
    names = ["o"]
    k1 = g._cache_key("t", names, {"o": np.array([0.10000001, 0.2, 0.9])})
    k2 = g._cache_key("t", names, {"o": np.array([0.10000009, 0.2, 0.9])})
    assert k1 == k2


def test_cache_key_handles_missing_positions():
    g = VLMObstacleGrounder(use_vlm=False)
    assert isinstance(g._cache_key("t", ["a", "b"], None), str)


# --- S3: setup_episode must clear per-episode vision state ---------------

def test_setup_episode_clears_vision_targets():
    f = FOLSafetyFilter()
    f._vision_target_xy = (0.1, 0.2)
    f._vision_targets_xy = [(0.1, 0.2), (0.3, 0.4)]
    f.setup_episode(obs={}, task_description="pick up the bowl")
    assert f._vision_target_xy is None
    assert f._vision_targets_xy == []


def test_setup_episode_evicts_stale_grounder_cache_entries():
    """A previous episode's entry must not survive into this one.

    The cache is legitimately repopulated by this episode's own grounding call
    (it still serves within-episode re-grounding at t=10), so the invariant is
    that the STALE key is gone -- not that the cache is empty.
    """
    f = FOLSafetyFilter()
    f.obstacle_grounder._cache["stale-key-from-episode-00"] = {
        "obstacle_obs_keys": ["wrong_obstacle"]
    }
    f.setup_episode(obs={}, task_description="pick up the bowl")
    assert "stale-key-from-episode-00" not in f.obstacle_grounder._cache, (
        "the grounder cache must not span episodes -- object poses are randomised"
    )
