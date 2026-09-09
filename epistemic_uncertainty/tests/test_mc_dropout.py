# epistemic_uncertainty/tests/test_mc_dropout.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from unittest.mock import MagicMock, patch
from epistemic_uncertainty.mc_dropout import MCDropoutEstimator


def _make_mock_model(action_dim=7, chunk=8):
    """Returns a mock that mimics vla.predict_action output."""
    model = MagicMock()
    model.train = MagicMock()
    model.eval = MagicMock()
    model.modules = MagicMock(return_value=[])
    model.predict_action = MagicMock(
        return_value=(np.random.randn(chunk, action_dim).astype(np.float32), None)
    )
    model.norm_stats = {}
    return model


def _make_dummy_obs():
    return {
        "full_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
    }


def test_mc_dropout_returns_correct_keys():
    model = _make_mock_model()
    cfg = MagicMock()
    cfg.model_family = "openvla"
    cfg.num_images_in_input = 1
    cfg.use_proprio = False
    cfg.use_film = False
    cfg.unnorm_key = "libero_spatial"

    estimator = MCDropoutEstimator(model=model, n_samples=5)

    obs = _make_dummy_obs()
    call_count = [0]
    def fake_get_action(*args, **kwargs):
        call_count[0] += 1
        rng = np.random.default_rng(call_count[0])
        chunk = rng.standard_normal((8, 7)).astype(np.float32)
        return [chunk[i] for i in range(8)]

    with patch("epistemic_uncertainty.mc_dropout.get_action", side_effect=fake_get_action):
        result = estimator.estimate(observation=obs, step=0,
                                    cfg=cfg, task_description="pick bowl",
                                    processor=None, action_head=None,
                                    proprio_projector=None, noisy_action_projector=None)

    assert result.method == "mc_dropout"
    assert result.step == 0
    for key in ("mean_action", "action_variance", "action_std", "trans_var_norm", "rot_var_norm"):
        assert key in result.values, f"Missing key: {key}"
    assert len(result.values["mean_action"]) == 7
    assert call_count[0] == 5


def test_mc_dropout_reset_clears_state():
    model = _make_mock_model()
    estimator = MCDropoutEstimator(model=model, n_samples=3)
    estimator.reset()  # should not raise


def test_mc_dropout_variance_nonzero_with_varying_samples():
    """With different actions per sample, variance must be > 0."""
    model = _make_mock_model()
    cfg = MagicMock()
    cfg.use_film = False

    estimator = MCDropoutEstimator(model=model, n_samples=4)
    obs = _make_dummy_obs()

    rng = np.random.default_rng(42)
    def varying_get_action(*args, **kwargs):
        chunk = rng.standard_normal((8, 7)).astype(np.float32)
        return [chunk[i] for i in range(8)]

    with patch("epistemic_uncertainty.mc_dropout.get_action", side_effect=varying_get_action):
        result = estimator.estimate(observation=obs, step=0,
                                    cfg=cfg, task_description="task",
                                    processor=None, action_head=None,
                                    proprio_projector=None, noisy_action_projector=None)

    assert result.values["trans_var_norm"] > 0
