# epistemic_uncertainty/tests/test_manager.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from unittest.mock import MagicMock
from epistemic_uncertainty.base import UncertaintyEstimate
from epistemic_uncertainty.manager import UncertaintyManager


def _mock_estimator(method_name, return_values):
    est = MagicMock()
    est.estimate.return_value = UncertaintyEstimate(
        method=method_name, step=0, values=return_values
    )
    est.reset = MagicMock()
    return est


def test_manager_aggregates_all_enabled():
    mc = _mock_estimator("mc_dropout", {"trans_var_norm": 0.1})
    ood = _mock_estimator("density_ood", {"is_ood": False})
    mgr = UncertaintyManager(estimators={"mc_dropout": mc, "density_ood": ood})

    obs = {"full_image": np.zeros((256, 256, 3), dtype=np.uint8)}
    result = mgr.estimate_step(observation=obs, step=0)

    assert "mc_dropout" in result
    assert "density_ood" in result
    assert result["mc_dropout"]["trans_var_norm"] == 0.1
    assert result["density_ood"]["is_ood"] is False


def test_manager_resets_all_estimators():
    e1 = _mock_estimator("mc_dropout", {})
    e2 = _mock_estimator("entropy", {})
    mgr = UncertaintyManager(estimators={"mc_dropout": e1, "entropy": e2})
    mgr.reset_episode()
    e1.reset.assert_called_once()
    e2.reset.assert_called_once()


def test_manager_empty_estimators():
    mgr = UncertaintyManager(estimators={})
    obs = {"full_image": np.zeros((256, 256, 3), dtype=np.uint8)}
    result = mgr.estimate_step(observation=obs, step=5)
    assert result == {}


def test_manager_skips_failed_estimator_and_records_error():
    bad = MagicMock()
    bad.estimate.side_effect = RuntimeError("GPU OOM")
    bad.reset = MagicMock()
    mgr = UncertaintyManager(estimators={"bad_method": bad})

    obs = {"full_image": np.zeros((256, 256, 3), dtype=np.uint8)}
    result = mgr.estimate_step(observation=obs, step=2)
    assert "bad_method" in result
    assert "error" in result["bad_method"]
