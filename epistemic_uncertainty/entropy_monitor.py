"""
Entropy-based uncertainty monitor.

Since OpenVLA-OFT uses an L1 regression action head, true token probabilities
are not available. We construct a proxy from MC-dropout action samples:
1. Compute element-wise variance across N samples.
2. Normalize variance by its max, then softmax → relative uncertainty distribution.
3. Normalized Shannon entropy H(softmax(var/max)) / H_max → entropy_score in [0, 1].
   - 0 = all uncertainty concentrated in one action dim (low spread)
   - 1 = uncertainty spread uniformly across all action dims (high spread)
4. confidence = exp(-trans_var_norm / variance_scale)
   - Calibrated to empirical data: trans_var_norm ~1e-4 separates collision from safe.
   - 1 = certain (small variance), 0 = highly uncertain (large variance).
"""

from typing import Any, Dict, List, Optional

import numpy as np

from epistemic_uncertainty.base import BaseUncertaintyEstimator, UncertaintyEstimate

_FAILURE_THRESHOLD = 0.3
# Empirically derived: trans_var_norm ~1e-4 separates collision/failure from success.
_DEFAULT_VARIANCE_SCALE = 1e-4


class EntropyMonitor(BaseUncertaintyEstimator):
    """Entropy proxy from action-sample variance distribution.

    Args:
        variance_scale: Scale for exponential confidence decay over trans_var_norm.
            At trans_var_norm == variance_scale, confidence ≈ 0.37.
    """

    def __init__(self, variance_scale: float = _DEFAULT_VARIANCE_SCALE) -> None:
        self.variance_scale = variance_scale

    def reset(self) -> None:
        pass

    def estimate_from_samples(
        self, action_samples: List[np.ndarray], step: int
    ) -> UncertaintyEstimate:
        """Compute entropy estimate given pre-collected MC dropout samples."""
        samples = np.stack(action_samples)  # (N, action_dim)
        variance = samples.var(axis=0)      # (action_dim,)

        var_max = float(variance.max())

        if var_max < 1e-10:
            # All samples identical → perfect confidence, zero entropy spread
            entropy_score = 0.0
            confidence = 1.0
        else:
            # Normalize variance to [0,1] before softmax so entropy reflects
            # the *shape* of uncertainty (which dims are uncertain), not magnitude.
            normed = variance / var_max
            probs = self._softmax(normed)
            h_raw = self._entropy(probs)
            h_max = np.log(len(variance)) if len(variance) > 1 else 1.0
            entropy_score = float(np.clip(h_raw / h_max, 0.0, 1.0))

            # Confidence calibrated to translational variance magnitude.
            trans_var_norm = float(np.linalg.norm(variance[:3]))
            confidence = float(np.exp(-trans_var_norm / self.variance_scale))

        label = "potential_failure" if confidence < _FAILURE_THRESHOLD else "nominal"

        return UncertaintyEstimate(
            method="entropy",
            step=step,
            values={
                "entropy_score": entropy_score,
                "confidence": confidence,
                "predicted_failure_label": label,
            },
        )

    def estimate(
        self,
        observation: Dict[str, Any],
        step: int,
        action_samples: Optional[List[np.ndarray]] = None,
        **_kwargs: Any,
    ) -> UncertaintyEstimate:
        """BaseUncertaintyEstimator interface — delegates to estimate_from_samples."""
        if action_samples is None or len(action_samples) == 0:
            return UncertaintyEstimate(
                method="entropy",
                step=step,
                values={"entropy_score": 0.0, "confidence": 1.0, "predicted_failure_label": "nominal"},
            )
        return self.estimate_from_samples(action_samples, step)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - x.max()
        exp_x = np.exp(x_shifted)
        return exp_x / exp_x.sum()

    def _entropy(self, probs: np.ndarray) -> float:
        safe = np.clip(probs, 1e-12, None)
        return float(-np.sum(safe * np.log(safe)))
