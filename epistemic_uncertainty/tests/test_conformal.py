# epistemic_uncertainty/tests/test_conformal.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from epistemic_uncertainty.conformal import ConformalPredictor


def test_calibrate_and_threshold():
    cal = ConformalPredictor(alpha=0.1)
    rng = np.random.default_rng(42)
    cal_scores = rng.standard_normal(100).tolist()
    cal.calibrate(cal_scores)
    # threshold should be near the 90th percentile of cal_scores
    expected = float(np.percentile(cal_scores, 90))
    assert cal.threshold == pytest.approx(expected, rel=0.05)


def test_predict_trust():
    cal = ConformalPredictor(alpha=0.1)
    cal.calibrate([float(i) for i in range(10)])  # scores 0..9
    # threshold ≈ 90th percentile ≈ 8.1
    result = cal.predict(score=5.0, step=3, ground_truth_outcome=None)
    assert result.values["decision"] == "trust"
    assert result.values["conformal_threshold"] > 0


def test_predict_abstain():
    cal = ConformalPredictor(alpha=0.1)
    cal.calibrate([float(i) for i in range(10)])
    result = cal.predict(score=9.5, step=7, ground_truth_outcome=None)
    assert result.values["decision"] == "abstain"


def test_predict_with_ground_truth():
    cal = ConformalPredictor(alpha=0.2)
    cal.calibrate([1.0, 2.0, 3.0, 4.0, 5.0])
    result = cal.predict(score=4.5, step=0, ground_truth_outcome="failure")
    assert "correct_decision" in result.values
    # abstain when failure → correct decision
    assert result.values["correct_decision"] is True


def test_not_calibrated_raises():
    cal = ConformalPredictor(alpha=0.1)
    with pytest.raises(RuntimeError, match="calibrate"):
        cal.predict(score=1.0, step=0)


def test_reset_clears_calibration():
    cal = ConformalPredictor(alpha=0.1)
    cal.calibrate([1.0, 2.0, 3.0])
    cal.reset()
    with pytest.raises(RuntimeError):
        cal.predict(score=1.0, step=0)
