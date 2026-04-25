"""
Entropy-based uncertainty monitor.

Since OpenVLA-OFT uses an L1 regression action head, true token probabilities
are not available. We construct a proxy from MC-dropout action samples:
1. Compute element-wise variance across N samples.
2. Softmax over variance vector → pseudo-probability distribution.
3. Shannon entropy H(softmax(var)).
4. confidence = 1 - H / H_max  (H_max = log(action_dim))
"""

from typing import Any, Dict, List, Optional

import numpy as np

from epistemic_uncertainty.base import BaseUncertaintyEstimator, UncertaintyEstimate

_FAILURE_THRESHOLD = 0.5


class EntropyMonitor(BaseUncertaintyEstimator):
    """Entropy proxy from action-sample variance distribution."""

    def reset(self) -> None:
        pass

    def estimate_from_samples(
        self, action_samples: List[np.ndarray], step: int
    ) -> UncertaintyEstimate:
        """Compute entropy estimate given pre-collected MC dropout samples."""
        samples = np.stack(action_samples)  # (N, action_dim)
        variance = samples.var(axis=0)      # (action_dim,)

        # Handle zero variance (all samples identical) → perfect confidence
        if np.allclose(variance, 0.0, atol=1e-12):
            entropy = 0.0
            confidence = 1.0
        else:
            probs = self._softmax(variance)
            entropy = self._entropy(probs)
            h_max = np.log(len(variance)) if len(variance) > 1 else 1.0
            confidence = float(1.0 - np.clip(entropy / h_max, 0.0, 1.0))

        label = "potential_failure" if confidence < _FAILURE_THRESHOLD else "nominal"

        return UncertaintyEstimate(
            method="entropy",
            step=step,
            values={
                "entropy_score": float(entropy),
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
