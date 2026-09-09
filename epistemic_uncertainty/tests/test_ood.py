# epistemic_uncertainty/tests/test_ood.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from unittest.mock import MagicMock
from epistemic_uncertainty.density_ood import DensityOODEstimator


def _make_feature_extractor(feat_dim=384):
    extractor = MagicMock()
    rng = np.random.default_rng(0)
    extractor.extract = MagicMock(
        side_effect=lambda img: rng.standard_normal(feat_dim).astype(np.float32)
    )
    return extractor


def test_fit_and_score_mahalanobis():
    extractor = _make_feature_extractor(feat_dim=16)
    est = DensityOODEstimator(feature_extractor=extractor, method="mahalanobis", ood_threshold=None)

    rng = np.random.default_rng(1)
    in_dist_images = [rng.standard_normal((256, 256, 3)).astype(np.uint8) for _ in range(50)]
    est.fit(in_dist_images)

    obs = {"full_image": rng.standard_normal((256, 256, 3)).astype(np.uint8)}
    result = est.estimate(observation=obs, step=2)

    assert result.method == "density_ood"
    assert result.step == 2
    for key in ("ood_score", "nn_distance", "is_ood"):
        assert key in result.values, f"Missing key: {key}"
    assert result.values["ood_score"] >= 0.0
    assert isinstance(result.values["is_ood"], bool)


def test_fit_and_score_knn():
    extractor = _make_feature_extractor(feat_dim=16)
    est = DensityOODEstimator(feature_extractor=extractor, method="knn", ood_threshold=10.0)

    rng = np.random.default_rng(2)
    in_dist_images = [rng.standard_normal((256, 256, 3)).astype(np.uint8) for _ in range(30)]
    est.fit(in_dist_images)

    obs = {"full_image": rng.standard_normal((256, 256, 3)).astype(np.uint8)}
    result = est.estimate(observation=obs, step=0)
    assert result.values["ood_score"] >= 0.0


def test_not_fitted_raises():
    extractor = _make_feature_extractor()
    est = DensityOODEstimator(feature_extractor=extractor, method="mahalanobis")
    obs = {"full_image": np.zeros((256, 256, 3), dtype=np.uint8)}
    with pytest.raises(RuntimeError, match="fit"):
        est.estimate(observation=obs, step=0)


def test_reset_does_not_clear_fit():
    extractor = _make_feature_extractor(feat_dim=8)
    est = DensityOODEstimator(feature_extractor=extractor, method="knn", ood_threshold=5.0)
    rng = np.random.default_rng(3)
    images = [rng.standard_normal((256, 256, 3)).astype(np.uint8) for _ in range(20)]
    est.fit(images)
    est.reset()  # reset per-episode state, not the fitted model
    obs = {"full_image": rng.standard_normal((256, 256, 3)).astype(np.uint8)}
    result = est.estimate(observation=obs, step=0)  # should not raise
    assert "ood_score" in result.values
