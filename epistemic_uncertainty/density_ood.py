"""
Density-based OOD detection.

Two backend methods:
  - "mahalanobis": fit mean + inverse covariance on in-distribution features.
      score = sqrt((f - mu)^T Sigma^{-1} (f - mu))
  - "knn": fit kNN index; score = distance to k-th nearest in-distribution neighbour.

Concrete feature extractors:
  - DINOv2FeatureExtractor: uses facebook/dinov2-base
  - VLABackboneExtractor: uses OpenVLA vision_backbone submodule
"""

from typing import Any, Dict, List, Optional

import numpy as np

from epistemic_uncertainty.base import BaseUncertaintyEstimator, UncertaintyEstimate


class DensityOODEstimator(BaseUncertaintyEstimator):
    """Mahalanobis or kNN OOD scorer on image features."""

    def __init__(
        self,
        feature_extractor: Any,
        method: str = "mahalanobis",
        ood_threshold: Optional[float] = None,
        knn_k: int = 5,
    ) -> None:
        assert method in ("mahalanobis", "knn"), f"Unknown method: {method}"
        self.extractor = feature_extractor
        self.method = method
        self.ood_threshold = ood_threshold
        self.knn_k = knn_k
        self._fitted = False
        self._mean: Optional[np.ndarray] = None
        self._inv_cov: Optional[np.ndarray] = None
        self._ref_features: Optional[np.ndarray] = None

    def fit(self, in_distribution_images: List[np.ndarray]) -> None:
        """Compute reference statistics from in-distribution images."""
        features = np.stack([self.extractor.extract(img) for img in in_distribution_images])
        self._ref_features = features

        if self.method == "mahalanobis":
            self._mean = features.mean(axis=0)
            cov = np.cov(features.T) + 1e-6 * np.eye(features.shape[1])
            self._inv_cov = np.linalg.inv(cov)

        # Auto-calibrate threshold at 95th percentile of in-dist scores
        if self.ood_threshold is None:
            in_scores = [self._compute_score(f) for f in features]
            self.ood_threshold = float(np.percentile(in_scores, 95))

        self._fitted = True

    def reset(self) -> None:
        pass  # fitted model is episode-independent; don't clear it

    def estimate(self, observation: Dict[str, Any], step: int, **_kwargs: Any) -> UncertaintyEstimate:
        if not self._fitted:
            raise RuntimeError("DensityOODEstimator must be .fit() before calling .estimate()")

        img = observation["full_image"]
        feat = self.extractor.extract(img)
        score = self._compute_score(feat)
        nn_distance = self._nn_distance(feat)

        return UncertaintyEstimate(
            method="density_ood",
            step=step,
            values={
                "ood_score": float(score),
                "nn_distance": float(nn_distance),
                "is_ood": bool(score > self.ood_threshold),
            },
        )

    def _compute_score(self, feat: np.ndarray) -> float:
        if self.method == "mahalanobis":
            diff = feat - self._mean
            return float(np.sqrt(diff @ self._inv_cov @ diff))
        else:  # knn
            return self._nn_distance(feat)

    def _nn_distance(self, feat: np.ndarray) -> float:
        if self._ref_features is None:
            return 0.0
        dists = np.linalg.norm(self._ref_features - feat, axis=1)
        k = min(self.knn_k, len(dists))
        return float(np.sort(dists)[k - 1])


class DINOv2FeatureExtractor:
    """Extracts CLS token from facebook/dinov2-base."""

    def __init__(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    def extract(self, image: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image
        pil_img = Image.fromarray(image)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().float().numpy()


class VLABackboneExtractor:
    """Extracts visual features from the OpenVLA vision_backbone submodule."""

    def __init__(self, vla_model: Any, device: str = "cuda") -> None:
        self.backbone = vla_model.vision_backbone
        self.device = device

    def extract(self, image: np.ndarray) -> np.ndarray:
        import torch
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        with torch.no_grad():
            feats = self.backbone(tensor)
        return feats.mean(dim=(1, 2, 3)).squeeze(0).cpu().float().numpy() \
            if feats.ndim == 4 else feats.squeeze(0).cpu().float().numpy()
