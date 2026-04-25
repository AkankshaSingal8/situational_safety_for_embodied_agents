"""
UncertaintyManager: coordinates all enabled uncertainty estimators.

Estimator failures are caught and recorded so a single failing method
never crashes the evaluation loop.
"""

import logging
from typing import Any, Dict

from epistemic_uncertainty.base import BaseUncertaintyEstimator

logger = logging.getLogger(__name__)


class UncertaintyManager:
    """Thin coordinator for a set of BaseUncertaintyEstimator instances."""

    def __init__(self, estimators: Dict[str, BaseUncertaintyEstimator]) -> None:
        self.estimators = estimators

    def estimate_step(
        self,
        observation: Dict[str, Any],
        step: int,
        **shared_kwargs: Any,
    ) -> Dict[str, Any]:
        """Call each estimator and return merged {method_name: values} dict."""
        results: Dict[str, Any] = {}
        for name, est in self.estimators.items():
            try:
                estimate = est.estimate(observation=observation, step=step, **shared_kwargs)
                results[name] = estimate.values
            except Exception as exc:
                logger.warning(f"Estimator '{name}' failed at step {step}: {exc}")
                results[name] = {"error": str(exc)}
        return results

    def reset_episode(self) -> None:
        for est in self.estimators.values():
            est.reset()
