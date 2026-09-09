"""
MC Dropout uncertainty estimator for OpenVLA action predictions.

get_vla_action() uses torch.inference_mode() which disables dropout.
We bypass it by calling get_action() after setting only Dropout submodules
to train() mode while keeping everything else in eval() mode.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiments.robot.robot_utils import get_action  # noqa: E402

from epistemic_uncertainty.base import BaseUncertaintyEstimator, UncertaintyEstimate


class MCDropoutEstimator(BaseUncertaintyEstimator):
    """Estimates action uncertainty via N stochastic MC-dropout forward passes.

    Args:
        trans_var_threshold: trans_var_norm above this triggers high_uncertainty=True.
            Default 1e-4 empirically separates collision/failure from success episodes.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 10,
        trans_var_threshold: float = 1e-4,
    ) -> None:
        self.model = model
        self.n_samples = n_samples
        self.trans_var_threshold = trans_var_threshold

    def reset(self) -> None:
        pass  # stateless between episodes

    def estimate(
        self,
        observation: Dict[str, Any],
        step: int,
        cfg: Any = None,
        task_description: str = "",
        processor: Any = None,
        action_head: Optional[nn.Module] = None,
        proprio_projector: Optional[nn.Module] = None,
        noisy_action_projector: Optional[nn.Module] = None,
        **_kwargs: Any,
    ) -> UncertaintyEstimate:
        """Run N dropout-enabled forward passes and compute action statistics."""
        # Enable only Dropout submodules (leave LayerNorm / BN in eval)
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

        action_samples: List[np.ndarray] = []
        try:
            for _ in range(self.n_samples):
                with torch.no_grad():
                    actions = get_action(
                        cfg,
                        self.model,
                        observation,
                        task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                    )
                action_samples.append(np.array(actions[0], dtype=np.float64))
        finally:
            self.model.eval()

        samples = np.stack(action_samples)  # (n_samples, action_dim)
        mean_action = samples.mean(axis=0)
        variance = samples.var(axis=0)
        std = samples.std(axis=0)

        trans_var_norm = float(np.linalg.norm(variance[:3]))
        rot_var_norm = float(np.linalg.norm(variance[3:6]))
        # dropout_active=False means all MC samples were identical (no dropout effect).
        dropout_active = bool(variance.max() > 1e-10)

        return UncertaintyEstimate(
            method="mc_dropout",
            step=step,
            values={
                "mean_action": mean_action.tolist(),
                "action_variance": variance.tolist(),
                "action_std": std.tolist(),
                "trans_var_norm": trans_var_norm,
                "rot_var_norm": rot_var_norm,
                "high_uncertainty": bool(trans_var_norm > self.trans_var_threshold),
                "dropout_active": dropout_active,
            },
        )
