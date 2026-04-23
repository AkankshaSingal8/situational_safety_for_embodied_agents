# /ocean/projects/cis250185p/asingal/test_run_episode_structure.py
"""
Structural tests for the per-chunk VLM architecture in run_episode() and run_task().
These tests verify the control flow and wiring without running the actual sim.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_run_task_signature_has_no_task_arg_removed():
    """run_task() must still accept all its existing args."""
    from run_libero_eval_integrated import run_task
    sig = inspect.signature(run_task)
    for param in ["cfg", "task_suite", "task_id", "model", "resize_size"]:
        assert param in sig.parameters, f"run_task() missing param: {param}"


def test_run_episode_signature_has_camera_specs():
    """run_episode() must accept camera_specs parameter."""
    from run_libero_eval_integrated import run_episode
    sig = inspect.signature(run_episode)
    assert "camera_specs" in sig.parameters, (
        "run_episode() must accept camera_specs for per-chunk obs capture"
    )


def test_generate_config_has_vlm_server_url():
    """GenerateConfig must have vlm_server_url field."""
    from run_libero_eval_integrated import GenerateConfig
    cfg = GenerateConfig()
    assert hasattr(cfg, "vlm_server_url"), "GenerateConfig missing vlm_server_url"
    assert "localhost" in cfg.vlm_server_url


def test_call_vlm_server_importable():
    """call_vlm_server must be importable (added in Task 4)."""
    from run_libero_eval_integrated import call_vlm_server
    assert callable(call_vlm_server)


def test_capture_chunk_obs_from_env_importable():
    """capture_chunk_obs_from_env must be importable (added in Task 4)."""
    from run_libero_eval_integrated import capture_chunk_obs_from_env
    assert callable(capture_chunk_obs_from_env)


def test_no_get_episode_cbf_constraints_call_in_run_episode_source():
    """run_episode() source must NOT call get_episode_cbf_constraints().
    That function was used for once-per-episode VLM; per-chunk uses call_vlm_server().
    """
    import ast
    with open(os.path.join(os.path.dirname(__file__), "run_libero_eval_integrated.py")) as f:
        source = f.read()
    tree = ast.parse(source)

    # Find run_episode function node
    run_episode_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_episode":
            run_episode_node = node
            break
    assert run_episode_node is not None, "run_episode() not found"

    # Check that get_episode_cbf_constraints is not called inside it
    for node in ast.walk(run_episode_node):
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            assert name != "get_episode_cbf_constraints", (
                "run_episode() must not call get_episode_cbf_constraints() — "
                "use call_vlm_server() per chunk instead"
            )


def test_call_vlm_server_called_in_run_episode_source():
    """run_episode() source must call call_vlm_server() for per-chunk VLM."""
    import ast
    with open(os.path.join(os.path.dirname(__file__), "run_libero_eval_integrated.py")) as f:
        source = f.read()
    tree = ast.parse(source)

    run_episode_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_episode":
            run_episode_node = node
            break
    assert run_episode_node is not None

    found = False
    for node in ast.walk(run_episode_node):
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "call_vlm_server":
                found = True
                break
    assert found, "run_episode() must call call_vlm_server() for per-chunk VLM"
