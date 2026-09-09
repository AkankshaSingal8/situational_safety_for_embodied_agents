"""
Split-conformal prediction for OpenVLA action trust/abstain decisions.

Usage:
  1. Run calibration episodes, collect per-step uncertainty scores.
  2. Call conformal.calibrate(scores) once.
  3. During evaluation, call conformal.predict(score, step) each model-query step.
"""

import math
from typing import Any, List, Optional

import numpy as np

from epistemic_uncertainty.base import BaseUncertaintyEstimator, UncertaintyEstimate


class ConformalPredictor(BaseUncertaintyEstimator):
    """Split-conformal predictor: calibrate once, predict per step."""

    def __init__(self, alpha: float = 0.1) -> None:
        assert 0 < alpha < 1, "alpha must be in (0, 1)"
        self.alpha = alpha
        self.threshold: Optional[float] = None
        self._calibrated = False

    def calibrate(self, nonconformity_scores: List[float]) -> None:
        """Fit conformal threshold from a list of calibration-set scores."""
        scores = np.array(nonconformity_scores, dtype=np.float64)
        n = len(scores)
        # Finite-sample corrected quantile: (n - ceil((n+1)*alpha) + 1) / n
        level = (n - math.ceil((n + 1) * self.alpha) + 1) / n
        level = max(0.0, min(level, 1.0))
        self.threshold = float(np.quantile(scores, level))
        self._calibrated = True

    def reset(self) -> None:
        """Clear calibration."""
        self.threshold = None
        self._calibrated = False

    def predict(
        self,
        score: float,
        step: int,
        ground_truth_outcome: Optional[str] = None,
    ) -> UncertaintyEstimate:
        """Return trust/abstain for the current uncertainty score."""
        if not self._calibrated:
            raise RuntimeError("Call calibrate() before predict()")

        decision = "abstain" if score > self.threshold else "trust"

        values: dict = {
            "uncertainty_score": float(score),
            "conformal_threshold": float(self.threshold),
            "decision": decision,
        }

        if ground_truth_outcome is not None:
            failure = ground_truth_outcome in ("failure", "collision", "unsafe")
            correct = (decision == "abstain") == failure
            values["ground_truth_outcome"] = ground_truth_outcome
            values["correct_decision"] = bool(correct)

        return UncertaintyEstimate(method="conformal", step=step, values=values)

    def estimate(
        self, observation: Any, step: int, score: float = 0.0, **_kwargs: Any
    ) -> UncertaintyEstimate:
        return self.predict(score=score, step=step)
