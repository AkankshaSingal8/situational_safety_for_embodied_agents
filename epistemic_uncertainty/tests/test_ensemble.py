# epistemic_uncertainty/tests/test_ensemble.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from unittest.mock import MagicMock, patch
from epistemic_uncertainty.deep_ensemble import DeepEnsembleEstimator


def _fake_action(seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(7).astype(np.float32) for _ in range(8)]


def test_ensemble_with_two_mock_checkpoints():
    checkpoints = ["/ckpt/a", "/ckpt/b"]

    call_count = [0]
    def fake_query(cfg, model, obs, task_desc, **kw):
        seed = call_count[0]
        call_count[0] += 1
        return _fake_action(seed)

    with patch("epistemic_uncertainty.deep_ensemble.get_action", side_effect=fake_query), \
         patch("epistemic_uncertainty.deep_ensemble._load_model_for_checkpoint",
               return_value=(MagicMock(), None, None, None, None)):
        cfg = MagicMock()
        cfg.use_film = False
        est = DeepEnsembleEstimator(
            base_cfg=cfg,
            ensemble_checkpoints=checkpoints,
        )
        obs = {"full_image": np.zeros((256, 256, 3), dtype=np.uint8),
               "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
               "state": np.zeros(8)}
        result = est.estimate(observation=obs, step=4, task_description="task")

    assert result.method == "deep_ensemble"
    assert result.step == 4
    for key in ("ensemble_mean", "ensemble_variance", "max_pairwise_disagreement", "pairwise_disagreements"):
        assert key in result.values, f"Missing key: {key}"
    assert len(result.values["ensemble_mean"]) == 7
    assert call_count[0] == 2


def test_ensemble_variance_zero_for_identical_models():
    checkpoints = ["/ckpt/a", "/ckpt/b", "/ckpt/c"]
    fixed_action = np.ones(7, dtype=np.float32)

    def identical_query(cfg, model, obs, task_desc, **kw):
        return [fixed_action.copy() for _ in range(8)]

    with patch("epistemic_uncertainty.deep_ensemble.get_action", side_effect=identical_query), \
         patch("epistemic_uncertainty.deep_ensemble._load_model_for_checkpoint",
               return_value=(MagicMock(), None, None, None, None)):
        cfg = MagicMock()
        cfg.use_film = False
        est = DeepEnsembleEstimator(base_cfg=cfg, ensemble_checkpoints=checkpoints)
        obs = {"full_image": np.zeros((256,256,3), dtype=np.uint8),
               "wrist_image": np.zeros((256,256,3), dtype=np.uint8),
               "state": np.zeros(8)}
        result = est.estimate(observation=obs, step=0, task_description="t")

    assert result.values["max_pairwise_disagreement"] == pytest.approx(0.0, abs=1e-6)


def test_ensemble_reset_is_idempotent():
    cfg = MagicMock()
    est = DeepEnsembleEstimator(base_cfg=cfg, ensemble_checkpoints=[])
    est.reset()
