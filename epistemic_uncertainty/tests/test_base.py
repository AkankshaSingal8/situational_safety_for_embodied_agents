# epistemic_uncertainty/tests/test_base.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from epistemic_uncertainty.base import UncertaintyEstimate, BaseUncertaintyEstimator


def test_uncertainty_estimate_creation():
    est = UncertaintyEstimate(method="mc_dropout", step=5, values={"variance": 0.1})
    assert est.method == "mc_dropout"
    assert est.step == 5
    assert est.values["variance"] == pytest.approx(0.1)


def test_uncertainty_estimate_to_dict():
    est = UncertaintyEstimate(method="entropy", step=0, values={"entropy_score": 1.2, "confidence": 0.8})
    d = est.to_dict()
    assert d["method"] == "entropy"
    assert d["values"]["entropy_score"] == pytest.approx(1.2)


def test_base_estimator_is_abstract():
    with pytest.raises(TypeError):
        BaseUncertaintyEstimator()


def test_concrete_estimator_must_implement_estimate():
    class PartialEstimator(BaseUncertaintyEstimator):
        def reset(self):
            pass
    with pytest.raises(TypeError):
        PartialEstimator()
