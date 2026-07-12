"""Static Context-Aware Viability Flow proof of concept.

This additive module shares the existing action-chunk filter interface without
changing the baseline evaluator or filter. It tests three parts of the proposed
architecture: route-conditioned residual repairs, segment-level hard checks,
and task-preserving candidate ranking. It contains no RL or online learning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence
import time

import numpy as np

from contextual_predictive_filter import (
    ContextProvider,
    FilterDecision,
    FilterMetrics,
    ObservationContextProvider,
    PredictiveFilterConfig,
    SafetyContext,
)


@dataclass
class CAVFStaticConfig(PredictiveFilterConfig):
    payload_radius: float = 0.015
    route_gain: float = 1.20
    route_sigma_fraction: float = 0.22
    endpoint_deviation_weight: float = 3.0
    correction_weight: float = 0.12
    smoothness_weight: float = 0.02
    route_switch_penalty: float = 0.015
    clearance_reward_weight: float = 1.0
    verifier_iterations: int = 5


@dataclass(frozen=True)
class CandidateDiagnostic:
    route: str
    hard_feasible: bool
    min_clearance: float
    endpoint_deviation: float
    correction_norm: float
    smoothness: float
    score: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CAVFFilterMetrics(FilterMetrics):
    route_counts: Dict[str, int] = field(default_factory=dict)
    candidate_diagnostics: list[list[Dict]] = field(default_factory=list)

    def summary(self) -> Dict:
        result = super().summary()
        result["route_counts"] = dict(self.route_counts)
        return result


class CAVFStaticSafetyFilter:
    """Static multimodal repair filter with a safe-success surrogate ranker."""

    _ROUTES = ("left", "right", "above")

    def __init__(
        self,
        config: Optional[PredictiveFilterConfig] = None,
        context_provider: Optional[ContextProvider] = None,
    ):
        # Existing evaluators build PredictiveFilterConfig. Preserve that
        # contract and add CAVF-only defaults without modifying evaluators.
        source = config or CAVFStaticConfig()
        values = {
            name: getattr(source, name)
            for name in PredictiveFilterConfig.__dataclass_fields__
            if hasattr(source, name)
        }
        self.config = CAVFStaticConfig(**values)
        self.context_provider = context_provider or ObservationContextProvider()
        self.context = SafetyContext("", ())
        self.metrics = CAVFFilterMetrics()
        self._active_route: Optional[str] = None

    def reset(self, obs: Dict, instruction: str, obstacle_name: Optional[str] = None) -> None:
        self.context = self.context_provider.build(obs, instruction, obstacle_name)
        self.metrics = CAVFFilterMetrics()
        self._active_route = None

    @staticmethod
    def _positions(start: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        return start[None, :] + np.cumsum(deltas, axis=0)

    @staticmethod
    def _segment_distance(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
        direction = end - start
        denominator = float(np.dot(direction, direction))
        if denominator <= 1e-12:
            return float(np.linalg.norm(point - start))
        fraction = float(np.clip(np.dot(point - start, direction) / denominator, 0.0, 1.0))
        return float(np.linalg.norm(point - (start + fraction * direction)))

    @staticmethod
    def _normalise(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        magnitude = float(np.linalg.norm(vector))
        return vector / magnitude if magnitude > 1e-8 else fallback.copy()

    def _required_clearance(self, hazard) -> float:
        cfg = self.config
        return hazard.radius + cfg.eef_radius + cfg.payload_radius + cfg.safety_margin

    def _segment_clearance(self, start: np.ndarray, positions: np.ndarray) -> float:
        if not self.context.hazards:
            return float("inf")
        previous = start
        result = float("inf")
        for current in positions:
            for hazard in self.context.hazards:
                result = min(
                    result,
                    self._segment_distance(previous, current, hazard.center)
                    - self._required_clearance(hazard),
                )
            previous = current
        return result

    def _closest_segment(self, start: np.ndarray, positions: np.ndarray, hazard) -> tuple[int, float]:
        previous = start
        best_index, best_distance = 0, float("inf")
        for index, current in enumerate(positions):
            distance = self._segment_distance(previous, current, hazard.center)
            if distance < best_distance:
                best_index, best_distance = index, distance
            previous = current
        return best_index, best_distance

    def _route_vector(self, route: str, start: np.ndarray, nominal: np.ndarray) -> np.ndarray:
        forward = nominal[-1] - start
        forward[2] = 0.0
        forward = self._normalise(forward, np.array([1.0, 0.0, 0.0]))
        left = np.array([-forward[1], forward[0], 0.0])
        return {
            "left": left,
            "right": -left,
            "above": np.array([0.0, 0.0, 1.0]),
        }[route]

    def _workspace_clip(self, positions: np.ndarray) -> np.ndarray:
        result = positions.copy()
        result[:, 2] = np.clip(
            result[:, 2], self.config.table_z_min, self.config.workspace_z_max
        )
        return result

    def _route_repair(self, start: np.ndarray, nominal: np.ndarray, route: str) -> np.ndarray:
        """Apply a smooth detour bump then verify every swept segment."""

        cfg = self.config
        repaired = nominal.copy()
        route_vector = self._route_vector(route, start, nominal)
        horizon = len(repaired)
        sigma = max(1.0, horizon * cfg.route_sigma_fraction)
        indices = np.arange(horizon, dtype=np.float64)

        for hazard in self.context.hazards:
            critical_index, critical_distance = self._closest_segment(start, nominal, hazard)
            required = self._required_clearance(hazard)
            if critical_distance >= required + cfg.influence_distance:
                continue
            amplitude = cfg.route_gain * (
                max(0.0, required - critical_distance) + cfg.influence_distance
            )
            bump = np.exp(-0.5 * ((indices - critical_index) / sigma) ** 2)
            repaired += bump[:, None] * amplitude * route_vector[None, :]

        repaired = self._workspace_clip(repaired)
        for _ in range(cfg.verifier_iterations):
            changed = False
            previous = start
            for index in range(len(repaired)):
                current = repaired[index]
                for hazard in self.context.hazards:
                    required = self._required_clearance(hazard)
                    distance = self._segment_distance(previous, current, hazard.center)
                    if distance >= required:
                        continue
                    midpoint = 0.5 * (previous + current)
                    radial = self._normalise(midpoint - hazard.center, route_vector)
                    correction = (
                        route_vector * cfg.route_gain * (required - distance + cfg.safety_margin)
                        + radial * (required - distance)
                    )
                    repaired[index:] += correction
                    repaired = self._workspace_clip(repaired)
                    changed = True
                previous = repaired[index]
            if not changed:
                break
        return repaired

    def _positions_to_commands(self, start: np.ndarray, positions: np.ndarray) -> np.ndarray:
        deltas = np.diff(np.vstack((start, positions)), axis=0)
        return np.clip(
            deltas / self.config.translation_scale,
            -self.config.max_translation_command,
            self.config.max_translation_command,
        )

    @staticmethod
    def _smoothness(commands: np.ndarray) -> float:
        return float(np.linalg.norm(np.diff(commands, axis=0))) if len(commands) > 1 else 0.0

    def _diagnostic(
        self,
        route: str,
        start: np.ndarray,
        nominal_positions: np.ndarray,
        candidate_positions: np.ndarray,
    ) -> CandidateDiagnostic:
        commands = self._positions_to_commands(start, candidate_positions)
        nominal_commands = self._positions_to_commands(start, nominal_positions)
        clearance = self._segment_clearance(start, candidate_positions)
        endpoint_deviation = float(np.linalg.norm(candidate_positions[-1] - nominal_positions[-1]))
        correction = float(np.linalg.norm(commands - nominal_commands))
        smoothness = self._smoothness(commands)
        feasible = clearance >= -1e-7
        score = (
            self.config.clearance_reward_weight * min(clearance, self.config.influence_distance)
            - self.config.endpoint_deviation_weight * endpoint_deviation
            - self.config.correction_weight * correction
            - self.config.smoothness_weight * smoothness
            - (
                self.config.route_switch_penalty
                if self._active_route is not None and route != self._active_route
                else 0.0
            )
        )
        return CandidateDiagnostic(
            route=route,
            hard_feasible=feasible,
            min_clearance=clearance,
            endpoint_deviation=endpoint_deviation,
            correction_norm=correction,
            smoothness=smoothness,
            score=score if feasible else -float("inf"),
        )

    def filter_chunk(self, actions: Sequence[Sequence[float]], obs: Dict) -> tuple[np.ndarray, FilterDecision]:
        started = time.perf_counter()
        chunk = np.asarray(actions, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] < 3:
            raise ValueError(f"Expected action chunk shaped (H,D>=3), got {chunk.shape}")
        if not np.all(np.isfinite(chunk)):
            raise ValueError("Action chunk contains non-finite values")

        start = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        nominal_commands = np.clip(
            chunk[:, :3],
            -self.config.max_translation_command,
            self.config.max_translation_command,
        )
        nominal_positions = self._positions(
            start, nominal_commands * self.config.translation_scale
        )
        nominal_clearance = self._segment_clearance(start, nominal_positions)
        candidates: list[tuple[str, np.ndarray]] = [("nominal", nominal_positions)]
        if nominal_clearance < self.config.influence_distance:
            candidates.extend(
                (route, self._route_repair(start, nominal_positions, route))
                for route in self._ROUTES
            )

        diagnostics = [
            self._diagnostic(route, start, nominal_positions, positions)
            for route, positions in candidates
        ]
        selected_index = int(np.argmax([item.score for item in diagnostics]))
        selected_route, selected_positions = candidates[selected_index]
        selected = diagnostics[selected_index]
        result = chunk.copy()
        if selected.hard_feasible:
            result[:, :3] = self._positions_to_commands(start, selected_positions)
        else:
            result[:, :3] = 0.0
            selected_route = "stop"

        correction = float(np.linalg.norm(result[:, :3] - chunk[:, :3]))
        intervened = correction > 1e-7
        if selected_route not in ("nominal", "stop"):
            self._active_route = selected_route
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        decision = FilterDecision(
            intervened=intervened,
            infeasible=not selected.hard_feasible,
            min_clearance_nominal=nominal_clearance,
            min_clearance_filtered=self._segment_clearance(
                start, self._positions(start, result[:, :3] * self.config.translation_scale)
            ),
            correction_norm=correction,
            latency_ms=elapsed_ms,
            reason=(
                "cavf_nominal_safe"
                if selected_route == "nominal"
                else "cavf_no_viable_route_stop"
                if selected_route == "stop"
                else f"cavf_viability_route_{selected_route}"
            ),
        )
        self.metrics.chunks += 1
        self.metrics.actions += len(chunk)
        self.metrics.interventions += int(intervened)
        self.metrics.infeasible += int(not selected.hard_feasible)
        self.metrics.correction_norm += correction
        self.metrics.latency_ms += elapsed_ms
        self.metrics.decisions.append(decision.to_dict())
        self.metrics.route_counts[selected_route] = self.metrics.route_counts.get(selected_route, 0) + 1
        self.metrics.candidate_diagnostics.append([item.to_dict() for item in diagnostics])
        return result.astype(np.asarray(actions).dtype), decision
