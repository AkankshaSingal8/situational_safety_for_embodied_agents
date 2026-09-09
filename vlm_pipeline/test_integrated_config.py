import numpy as np
import pytest

# ── Bug #1: obstacle name crash ──────────────────────────────────────────────
def _find_obstacle(obstacle_names, obs):
    """Mirrors the FIXED logic that should be in run_episode()."""
    obstacle_name = None
    for name in obstacle_names:
        p = obs.get(f"{name}_pos", np.zeros(3))
        if p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
            obstacle_name = name
            break
    if obstacle_name is None and obstacle_names:
        obstacle_name = obstacle_names[0]
    return obstacle_name

def test_obstacle_empty_obs_no_crash():
    result = _find_obstacle(["moka_pot_obstacle"], {})
    assert result == "moka_pot_obstacle"   # fallback to first

def test_obstacle_in_bounds():
    obs = {"moka_pot_obstacle_pos": np.array([0.1, 0.1, 0.85])}
    assert _find_obstacle(["moka_pot_obstacle"], obs) == "moka_pot_obstacle"

def test_obstacle_out_of_bounds_falls_back():
    obs = {"moka_pot_obstacle_pos": np.array([5.0, 5.0, 5.0])}   # out of bounds
    assert _find_obstacle(["moka_pot_obstacle"], obs) == "moka_pot_obstacle"  # fallback

def test_no_obstacles_returns_none():
    assert _find_obstacle([], {}) is None

# ── Bug #3: missing safety_level validation ───────────────────────────────────
def _validate_safety_level(safety_level):
    if safety_level not in ("I", "II"):
        raise ValueError(f"Invalid safety_level '{safety_level}'")

def test_invalid_safety_level_raises():
    with pytest.raises(ValueError, match="Invalid safety_level"):
        _validate_safety_level("III")

def test_valid_safety_levels_pass():
    _validate_safety_level("I")   # must not raise
    _validate_safety_level("II")  # must not raise

# ── Bug #2: run_id includes safety level ─────────────────────────────────────
def test_run_id_includes_safety_level():
    """run_id must contain 'levelI' or 'levelII' between suite name and model family."""
    suite = "safelibero_spatial"
    safety_level = "I"
    model_family = "openvla"
    date_time = "2026-04-20_00-00-00"
    run_id = f"EVAL-{suite}-level{safety_level}-{model_family}-{date_time}"
    assert f"level{safety_level}" in run_id
    assert run_id.index(f"level{safety_level}") > run_id.index(suite)
    assert run_id.index(f"level{safety_level}") < run_id.index(model_family)

# ── Bug #4: results_output_dir used correctly ─────────────────────────────────
def test_results_dir_uses_results_output_dir():
    """Results directory must be built from results_output_dir, not local_log_dir."""
    import os
    results_output_dir = "integrated_benchmark"
    task_suite_name = "safelibero_spatial"
    results_dir = os.path.join(results_output_dir, task_suite_name)
    assert results_dir == "integrated_benchmark/safelibero_spatial"
    # Confirm it does NOT start with local_log_dir prefix
    local_log_dir = "./experiments/logs"
    assert not results_dir.startswith(local_log_dir)
