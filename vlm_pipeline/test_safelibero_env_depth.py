import inspect
import pytest

def test_get_safelibero_env_has_camera_depths_param():
    """camera_depths parameter must exist with default False."""
    from safelibero_utils import get_safelibero_env
    sig = inspect.signature(get_safelibero_env)
    assert "camera_depths" in sig.parameters, "Missing camera_depths param"
    assert sig.parameters["camera_depths"].default == False

def test_get_safelibero_env_has_camera_segmentations_param():
    """camera_segmentations parameter must exist with default None."""
    from safelibero_utils import get_safelibero_env
    sig = inspect.signature(get_safelibero_env)
    assert "camera_segmentations" in sig.parameters, "Missing camera_segmentations param"
    assert sig.parameters["camera_segmentations"].default is None

def test_existing_callers_unaffected():
    """Calling with old signature (no new args) must not raise TypeError."""
    from safelibero_utils import get_safelibero_env
    sig = inspect.signature(get_safelibero_env)
    # Old positional args: task, model_family, resolution, include_wrist_camera
    # New args must be keyword-only with defaults, so old callers still work
    params = list(sig.parameters.keys())
    assert params[0] == "task"
    assert params[1] == "model_family"
    # New params must have defaults (so old callers don't need to pass them)
    assert sig.parameters["camera_depths"].default is not inspect.Parameter.empty
    assert sig.parameters["camera_segmentations"].default is not inspect.Parameter.empty
