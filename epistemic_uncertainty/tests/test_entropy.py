# epistemic_uncertainty/tests/test_entropy.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from epistemic_uncertainty.entropy_monitor import EntropyMonitor


def test_entropy_uniform_distribution():
    """Uniform probs → max entropy."""
    monitor = EntropyMonitor()
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    h = monitor._entropy(probs)
    assert h == pytest.approx(np.log(4), abs=1e-6)


def test_entropy_deterministic_distribution():
    """One-hot probs → zero entropy."""
    monitor = EntropyMonitor()
    probs = np.array([1.0, 0.0, 0.0, 0.0])
    h = monitor._entropy(probs)
    assert h == pytest.approx(0.0, abs=1e-6)


def test_estimate_from_action_samples():
    """Estimate should return correct keys and bounded values."""
    monitor = EntropyMonitor()
    rng = np.random.default_rng(0)
    action_samples = [rng.standard_normal(7).astype(np.float32) for _ in range(5)]
    result = monitor.estimate_from_samples(action_samples=action_samples, step=3)

    assert result.method == "entropy"
    assert result.step == 3
    for key in ("entropy_score", "confidence", "predicted_failure_label"):
        assert key in result.values, f"Missing key: {key}"
    assert 0.0 <= result.values["confidence"] <= 1.0
    assert result.values["entropy_score"] >= 0.0


def test_estimate_from_samples_constant_actions():
    """Identical action samples → zero entropy → max confidence."""
    monitor = EntropyMonitor()
    action = np.ones(7, dtype=np.float32)
    action_samples = [action.copy() for _ in range(5)]
    result = monitor.estimate_from_samples(action_samples=action_samples, step=0)
    assert result.values["confidence"] == pytest.approx(1.0, abs=1e-4)


def test_estimate_reset_is_idempotent():
    monitor = EntropyMonitor()
    monitor.reset()
    monitor.reset()
