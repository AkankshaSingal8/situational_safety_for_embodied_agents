"""Policy-agnostic predictive safety filter for Cartesian action chunks.

The filter deliberately knows nothing about a VLA implementation.  A policy adapter
provides an ``(H, 7)`` action chunk and the filter returns a chunk with the same
shape.  The first three action dimensions are interpreted as Cartesian EEF deltas;
rotation and gripper commands are preserved.

For the SafeLIBERO pilot, context is supplied by ``ObservationContextProvider``.
It reads named object poses from the standard LIBERO observation dictionary.  The
provider is isolated from the projector so it can later be replaced by RGB-D/VLM
grounding without changing policy integrations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Optional, Protocol, Sequence
import time

import numpy as np


@dataclass(frozen=True)
class Hazard:
    name: str
    center: np.ndarray
    radius: float
    kind: str = "collision"


@dataclass(frozen=True)
class SafetyContext:
    instruction: str
    hazards: tuple[Hazard, ...]


class ContextProvider(Protocol):
    def build(self, obs: Dict, instruction: str, obstacle_name: Optional[str]) -> SafetyContext:
        ...


class ObservationContextProvider:
    """SafeLIBERO context adapter; replaceable by an RGB-D/VLM provider.

    Object names determine conservative envelope sizes.  Simulator object state is
    used only in this pilot adapter and is reported explicitly in result metadata.
    """

    _RADII = {
        "moka": 0.080,
        "wine": 0.060,
        "bottle": 0.060,
        "milk": 0.065,
        "carton": 0.065,
        "box": 0.070,
        "book": 0.065,
        "mug": 0.060,
    }

    def __init__(self, default_radius: float = 0.065):
        self.default_radius = float(default_radius)

    def _radius(self, name: str) -> float:
        label = name.lower().replace("_", " ")
        return next((radius for key, radius in self._RADII.items() if key in label), self.default_radius)

    def build(self, obs: Dict, instruction: str, obstacle_name: Optional[str]) -> SafetyContext:
        hazards = []
        names: Iterable[str]
        if obstacle_name:
            names = (obstacle_name,)
        else:
            names = (key[:-4] for key in obs if key.endswith("_pos") and "obstacle" in key)
        for name in names:
            pos = obs.get(f"{name}_pos")
            if pos is None:
                continue
            center = np.asarray(pos, dtype=np.float64)
            if center.shape != (3,):
                continue
            hazards.append(Hazard(name=name, center=center.copy(), radius=self._radius(name)))
        return SafetyContext(instruction=instruction, hazards=tuple(hazards))


@dataclass
class PredictiveFilterConfig:
    safety_margin: float = 0.035
    eef_radius: float = 0.025
    max_translation_delta: float = 0.050
    max_correction_per_step: float = 0.035
    influence_distance: float = 0.055
    detour_gain: float = 0.75
    smoothing: float = 0.30
    table_z_min: float = 0.805
    workspace_z_max: float = 1.35
    numerical_eps: float = 1e-8


@dataclass
class FilterDecision:
    intervened: bool
    infeasible: bool
    min_clearance_nominal: float
    min_clearance_filtered: float
    correction_norm: float
    latency_ms: float
    reason: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FilterMetrics:
    chunks: int = 0
    actions: int = 0
    interventions: int = 0
    infeasible: int = 0
    correction_norm: float = 0.0
    latency_ms: float = 0.0
    decisions: list[Dict] = field(default_factory=list)

    def summary(self) -> Dict:
        return {
            "chunks": self.chunks,
            "actions": self.actions,
            "interventions": self.interventions,
            "intervention_rate": self.interventions / self.chunks if self.chunks else 0.0,
            "infeasible": self.infeasible,
            "mean_correction_norm": self.correction_norm / self.chunks if self.chunks else 0.0,
            "mean_latency_ms": self.latency_ms / self.chunks if self.chunks else 0.0,
        }


class ContextualPredictiveSafetyFilter:
    """Predict and minimally redirect an entire Cartesian action chunk."""

    def __init__(
        self,
        config: Optional[PredictiveFilterConfig] = None,
        context_provider: Optional[ContextProvider] = None,
    ):
        self.config = config or PredictiveFilterConfig()
        self.context_provider = context_provider or ObservationContextProvider()
        self.context = SafetyContext("", ())
        self.metrics = FilterMetrics()

    def reset(self, obs: Dict, instruction: str, obstacle_name: Optional[str] = None) -> None:
        self.context = self.context_provider.build(obs, instruction, obstacle_name)
        self.metrics = FilterMetrics()

    @staticmethod
    def _positions(start: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        return start[None, :] + np.cumsum(deltas, axis=0)

    def _clearance(self, positions: np.ndarray) -> float:
        if not self.context.hazards:
            return float("inf")
        values = []
        for hazard in self.context.hazards:
            required = hazard.radius + self.config.eef_radius + self.config.safety_margin
            values.extend(np.linalg.norm(positions - hazard.center[None, :], axis=1) - required)
        return float(np.min(values))

    def _project_positions(self, start: np.ndarray, nominal: np.ndarray) -> tuple[np.ndarray, bool]:
        cfg = self.config
        safe = nominal.copy()
        previous = start.copy()
        infeasible = False

        for i in range(len(safe)):
            desired = safe[i]
            modified = False
            for hazard in self.context.hazards:
                required = hazard.radius + cfg.eef_radius + cfg.safety_margin
                offset = desired - hazard.center
                distance = float(np.linalg.norm(offset))
                if distance >= required + cfg.influence_distance:
                    continue

                if distance < cfg.numerical_eps:
                    radial = previous - hazard.center
                    if np.linalg.norm(radial) < cfg.numerical_eps:
                        radial = np.array([1.0, 0.0, 0.0])
                else:
                    radial = offset
                radial = radial / max(np.linalg.norm(radial), cfg.numerical_eps)

                # Select a consistent horizontal tangent that advances the nominal
                # path instead of reflecting it back toward the start.
                tangent_a = np.array([-radial[1], radial[0], 0.0])
                if np.linalg.norm(tangent_a) < cfg.numerical_eps:
                    tangent_a = np.array([1.0, 0.0, 0.0])
                tangent_a /= np.linalg.norm(tangent_a)
                nominal_motion = nominal[min(i + 1, len(nominal) - 1)] - previous
                tangent = tangent_a if np.dot(tangent_a, nominal_motion) >= 0 else -tangent_a

                penetration = max(0.0, required - distance)
                proximity = max(0.0, required + cfg.influence_distance - distance)
                radial_push = radial * penetration
                tangential_push = tangent * cfg.detour_gain * proximity
                correction = radial_push + tangential_push
                correction_norm = np.linalg.norm(correction)
                if correction_norm > cfg.max_correction_per_step:
                    correction *= cfg.max_correction_per_step / correction_norm
                desired = desired + correction
                modified = True

                # Hard projection is the final safety backstop.
                final_offset = desired - hazard.center
                final_distance = np.linalg.norm(final_offset)
                if final_distance < required:
                    if final_distance < cfg.numerical_eps:
                        infeasible = True
                        desired = previous.copy()
                    else:
                        desired = hazard.center + final_offset / final_distance * required

            desired[2] = np.clip(desired[2], cfg.table_z_min, cfg.workspace_z_max)
            if modified and i and cfg.smoothing > 0:
                desired = (1.0 - cfg.smoothing) * desired + cfg.smoothing * safe[i - 1]
            safe[i] = desired
            previous = desired
        return safe, infeasible

    def filter_chunk(self, actions: Sequence[Sequence[float]], obs: Dict) -> tuple[np.ndarray, FilterDecision]:
        started = time.perf_counter()
        chunk = np.asarray(actions, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] < 3:
            raise ValueError(f"Expected action chunk shaped (H,D>=3), got {chunk.shape}")
        if not np.all(np.isfinite(chunk)):
            raise ValueError("Action chunk contains non-finite values")

        start = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        deltas = np.clip(
            chunk[:, :3], -self.config.max_translation_delta, self.config.max_translation_delta
        )
        nominal_positions = self._positions(start, deltas)
        nominal_clearance = self._clearance(nominal_positions)
        safe_positions, infeasible = self._project_positions(start, nominal_positions)

        safe_deltas = np.diff(np.vstack([start, safe_positions]), axis=0)
        safe_deltas = np.clip(
            safe_deltas, -self.config.max_translation_delta, self.config.max_translation_delta
        )
        result = chunk.copy()
        result[:, :3] = safe_deltas
        correction = float(np.linalg.norm(result[:, :3] - chunk[:, :3]))
        intervened = correction > 1e-7
        filtered_clearance = self._clearance(self._positions(start, result[:, :3]))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        decision = FilterDecision(
            intervened=intervened,
            infeasible=infeasible,
            min_clearance_nominal=nominal_clearance,
            min_clearance_filtered=filtered_clearance,
            correction_norm=correction,
            latency_ms=elapsed_ms,
            reason="predicted_hazard_intersection" if intervened else "nominal_chunk_safe",
        )
        self.metrics.chunks += 1
        self.metrics.actions += len(chunk)
        self.metrics.interventions += int(intervened)
        self.metrics.infeasible += int(infeasible)
        self.metrics.correction_norm += correction
        self.metrics.latency_ms += elapsed_ms
        self.metrics.decisions.append(decision.to_dict())
        return result.astype(np.asarray(actions).dtype), decision
