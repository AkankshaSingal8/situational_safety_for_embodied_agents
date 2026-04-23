# /ocean/projects/cis250185p/asingal/test_chunk_obs.py
"""Tests for call_vlm_server() and capture_chunk_obs_from_env()."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_call_vlm_server_dry_run_returns_dict():
    """dry_run=True must return a dict without hitting any server."""
    from run_libero_eval_integrated import call_vlm_server
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "out.json")
        result = call_vlm_server(
            obs_folder=tmpdir,
            output_json_path=out,
            server_url="http://localhost:5001",
            method="m1",
            dry_run=True,
        )
    assert result is not None, "dry_run should return a dict, not None"
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    # Must have 'single' key (matches DRY_RUN_RESPONSE schema)
    assert "single" in result, f"Expected 'single' key, got keys: {list(result.keys())}"


def test_call_vlm_server_connection_error_returns_none():
    """Connection error (nothing on port 9999) must return None, not raise."""
    from run_libero_eval_integrated import call_vlm_server
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "out.json")
        result = call_vlm_server(
            obs_folder=tmpdir,
            output_json_path=out,
            server_url="http://localhost:9999",
            method="m1",
            dry_run=False,
        )
    assert result is None, f"Connection error should return None, got {result}"


def test_call_vlm_server_dry_run_writes_json_file():
    """dry_run=True must write the result to output_json_path."""
    from run_libero_eval_integrated import call_vlm_server
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "result.json")
        call_vlm_server(
            obs_folder=tmpdir,
            output_json_path=out,
            server_url="http://localhost:5001",
            method="m1",
            dry_run=True,
        )
        assert os.path.exists(out), "output_json_path must be written"
        with open(out) as f:
            data = json.load(f)
        assert "single" in data


def test_capture_chunk_obs_from_env_function_exists():
    """capture_chunk_obs_from_env must be importable from run_libero_eval_integrated."""
    from run_libero_eval_integrated import capture_chunk_obs_from_env
    import inspect
    sig = inspect.signature(capture_chunk_obs_from_env)
    params = list(sig.parameters.keys())
    # Must accept obs, sim, obs_folder, task_id, episode_idx, chunk_idx,
    #              safety_level, task_description, resolution, camera_specs
    for required in ["obs", "sim", "obs_folder", "task_id", "episode_idx",
                     "chunk_idx", "safety_level", "task_description",
                     "resolution", "camera_specs"]:
        assert required in params, f"Missing param: {required}"


def test_call_vlm_server_function_exists_with_correct_signature():
    """call_vlm_server must be importable with correct signature."""
    from run_libero_eval_integrated import call_vlm_server
    import inspect
    sig = inspect.signature(call_vlm_server)
    params = sig.parameters
    assert "obs_folder" in params
    assert "output_json_path" in params
    assert "server_url" in params
    assert "method" in params
    assert "dry_run" in params
    # dry_run must default to False
    assert params["dry_run"].default == False
