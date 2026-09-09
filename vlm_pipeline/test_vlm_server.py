# /ocean/projects/cis250185p/asingal/test_vlm_server.py
"""
Tests for qwen_vlm_server.py that do NOT require starting the server or GPU.
Live server tests (requiring the qwen env + GPU) are in the plan's Task 6.
"""
import importlib
import os
import sys
import json
import types
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_server_module_importable_without_flask_error():
    """qwen_vlm_server.py must be importable (checks syntax + top-level structure)."""
    # We mock flask to avoid needing it in the test env
    flask_mock = types.ModuleType("flask")
    flask_mock.Flask = lambda *a, **kw: types.SimpleNamespace(
        route=lambda *a, **kw: (lambda f: f),
        run=lambda *a, **kw: None,
    )
    flask_mock.jsonify = lambda x: x
    flask_mock.request = types.SimpleNamespace(get_json=lambda **kw: {})
    sys.modules.setdefault("flask", flask_mock)

    # Also mock transformers to avoid GPU load
    if "transformers" not in sys.modules:
        sys.modules["transformers"] = types.ModuleType("transformers")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qwen_vlm_server",
        os.path.join(os.path.dirname(__file__), "qwen_vlm_server.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Should not raise
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass  # argparse --help exits; that's fine


def test_dry_run_result_schema():
    """DRY_RUN_RESULT must have correct schema."""
    from qwen_vlm_worker import DRY_RUN_RESULT
    assert "single" in DRY_RUN_RESULT
    episode = DRY_RUN_RESULT["single"]
    assert "description" in episode
    assert "end_object" in episode
    assert "objects" in episode
    assert isinstance(episode["objects"], list)


def test_qwen_models_dict_populated():
    """QWEN_MODELS must have at least one model key."""
    from qwen_vlm_worker import QWEN_MODELS
    assert len(QWEN_MODELS) >= 1
    for key, val in QWEN_MODELS.items():
        assert isinstance(key, str) and len(key) > 0
        assert isinstance(val, str) and "/" in val   # HuggingFace model ID


def test_server_file_has_health_and_infer_routes():
    """qwen_vlm_server.py source must contain /health and /infer route definitions."""
    server_path = os.path.join(os.path.dirname(__file__), "qwen_vlm_server.py")
    assert os.path.exists(server_path), "qwen_vlm_server.py not found"
    source = open(server_path).read()
    assert '"/health"' in source or "'/health'" in source, "Missing /health route"
    assert '"/infer"' in source or "'/infer'" in source, "Missing /infer route"


def test_server_file_has_load_model_function():
    """qwen_vlm_server.py must define a model loading function."""
    server_path = os.path.join(os.path.dirname(__file__), "qwen_vlm_server.py")
    source = open(server_path).read()
    assert "def _load_model" in source or "def load_model" in source


def test_server_file_imports_dry_run_result():
    """qwen_vlm_server.py must import DRY_RUN_RESULT from qwen_vlm_worker."""
    server_path = os.path.join(os.path.dirname(__file__), "qwen_vlm_server.py")
    source = open(server_path).read()
    assert "DRY_RUN_RESULT" in source
