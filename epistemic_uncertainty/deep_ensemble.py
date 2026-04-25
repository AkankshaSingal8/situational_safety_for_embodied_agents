"""
Deep Ensemble uncertainty estimator.

Loads each checkpoint sequentially, queries it, then frees GPU memory.
This avoids multi-model OOM at the cost of latency (acceptable for offline eval).
"""

import copy
import gc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from experiments.robot.robot_utils import get_action, get_model
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
)

from epistemic_uncertainty.base import BaseUncertaintyEstimator, UncertaintyEstimate


def _load_model_for_checkpoint(
    base_cfg: Any, checkpoint_path: str
) -> Tuple[Any, Any, Any, Any, Any]:
    """Load a fresh model + components for the given checkpoint path."""
    cfg = copy.deepcopy(base_cfg)
    cfg.pretrained_checkpoint = checkpoint_path

    model = get_model(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    return model, action_head, proprio_projector, noisy_action_projector, processor


class DeepEnsembleEstimator(BaseUncertaintyEstimator):
    """Queries each checkpoint in sequence, aggregates disagreement metrics."""

    def __init__(self, base_cfg: Any, ensemble_checkpoints: List[str]) -> None:
        self.base_cfg = base_cfg
        self.ensemble_checkpoints = ensemble_checkpoints

    def reset(self) -> None:
        pass

    def estimate(
        self,
        observation: Dict[str, Any],
        step: int,
        task_description: str = "",
        **_kwargs: Any,
    ) -> UncertaintyEstimate:
        first_actions: List[np.ndarray] = []

        for ckpt in self.ensemble_checkpoints:
            model, action_head, proprio_proj, noisy_proj, processor = \
                _load_model_for_checkpoint(self.base_cfg, ckpt)
            try:
                actions = get_action(
                    self.base_cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_proj,
                    noisy_action_projector=noisy_proj,
                    use_film=self.base_cfg.use_film,
                )
                first_actions.append(np.array(actions[0], dtype=np.float64))
            finally:
                del model, action_head, proprio_proj, noisy_proj, processor
                gc.collect()
                torch.cuda.empty_cache()

        if not first_actions:
            return UncertaintyEstimate(
                method="deep_ensemble", step=step,
                values={"ensemble_mean": [], "ensemble_variance": [],
                        "pairwise_disagreements": [], "max_pairwise_disagreement": 0.0},
            )

        stack = np.stack(first_actions)  # (K, action_dim)
        mean = stack.mean(axis=0)
        variance = stack.var(axis=0)

        K = len(first_actions)
        disagreements = []
        for i in range(K):
            for j in range(i + 1, K):
                disagreements.append(float(np.linalg.norm(first_actions[i] - first_actions[j])))

        return UncertaintyEstimate(
            method="deep_ensemble",
            step=step,
            values={
                "ensemble_mean": mean.tolist(),
                "ensemble_variance": variance.tolist(),
                "pairwise_disagreements": disagreements,
                "max_pairwise_disagreement": float(max(disagreements)) if disagreements else 0.0,
            },
        )
