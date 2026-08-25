"""
FOL Safety Filter — CBF Mapper

Translates a list of ActiveConstraint objects (from the KB evaluator) into
CBF envelopes that the existing SemanticSafetyFilter can consume.

The existing filter expects:
  - semantic_envelopes: List[(SuperquadricParams, alpha)]
  - collision_envelopes: List[(SuperquadricParams, alpha)]
  - pose_constrained: bool
  - velocity_limit: float | None  (new — handled here)

All code that depends on the existing SemanticSafetyFilter's internal API
lives here so filter.py stays clean.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# We reuse the superquadric type from semantic_cbf_filter.py.
# Import it dynamically so this module can be imported standalone for tests.
# ---------------------------------------------------------------------------

def _get_sq_class():
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from vlm_pipeline.semantic_cbf_filter import SuperquadricParams
        return SuperquadricParams
    except ImportError:
        # Fallback dataclass for standalone testing
        from dataclasses import dataclass as dc
        import numpy as _np

        @dc
        class _SQ:
            a1: float = 0.05; a2: float = 0.05; a3: float = 0.05
            e1: float = 1.0;  e2: float = 1.0
            position: _np.ndarray = field(default_factory=lambda: _np.zeros(3))
            rotation: _np.ndarray = field(default_factory=lambda: _np.eye(3))
        return _SQ


@dataclass
class MappedCBFSet:
    """Everything needed to reconfigure SemanticSafetyFilter for this timestep."""
    semantic_envelopes: List[Tuple]   # (SuperquadricParams, alpha)
    collision_envelopes: List[Tuple]  # (SuperquadricParams, alpha)
    pose_constrained: bool = False
    desired_orientation: Optional[np.ndarray] = None
    # velocity-limiting (new)
    velocity_limit: Optional[float] = None  # m/s; None = no limit
    angular_limit: Optional[float] = None   # rad/s; None = no limit
    tilt_max_rad: Optional[float] = None    # tilt lock
    # metadata
    active_rule_names: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Superquadric helpers (duplicated from semantic_cbf_filter.py to be
# importable without the full pipeline environment)
# ---------------------------------------------------------------------------

def _sphere_sq(center: np.ndarray, radius: float) -> object:
    """Axis-aligned sphere superquadric with safety margin."""
    SQ = _get_sq_class()
    return SQ(
        a1=radius, a2=radius, a3=radius,
        e1=1.0, e2=1.0,
        position=center.copy(),
        rotation=np.eye(3),
    )


def _box_sq(center: np.ndarray, half_extents: np.ndarray,
            margin: float = 0.02) -> object:
    """Box-shaped superquadric (high exponents ≈ box)."""
    SQ = _get_sq_class()
    a = np.maximum(half_extents, 0.03) + margin
    return SQ(
        a1=float(a[0]), a2=float(a[1]), a3=float(a[2]),
        e1=0.2, e2=0.2,
        position=center.copy(),
        rotation=np.eye(3),
    )


def _above_sq(obj_pos: np.ndarray, obj_bbox: np.ndarray,
              ws_z_max: float = 1.40) -> object:
    """Vertical half-space above the object (avoid reaching above it)."""
    SQ = _get_sq_class()
    height = max(ws_z_max - obj_pos[2], 0.10)
    center = obj_pos.copy()
    center[2] = obj_pos[2] + height / 2.0
    a = np.maximum(obj_bbox[:2], 0.05) + 0.04
    return SQ(
        a1=float(a[0]), a2=float(a[1]), a3=height / 2.0,
        e1=0.2, e2=0.2,
        position=center,
        rotation=np.eye(3),
    )


# ---------------------------------------------------------------------------
# Main mapper
# ---------------------------------------------------------------------------

class CBFMapper:
    """
    Builds a MappedCBFSet from a list of ActiveConstraints each timestep.

    Stateless: takes active constraints + current ee_quat → returns MappedCBFSet.
    """

    # Default parameters
    ALPHA_DEFAULT = 1.0
    ALPHA_CAUTION = 0.25
    ALPHA_FRAGILE = 0.15
    SAFETY_MARGIN = 0.025         # extra radius on collision envelopes
    SPATIAL_MARGIN = 0.04         # extra radius on semantic envelopes

    def __init__(self, ws_z_max: float = 1.40):
        self.ws_z_max = ws_z_max

    def map(self, active_constraints, ee_quat: np.ndarray) -> MappedCBFSet:
        """
        active_constraints: List[ActiveConstraint] from kb.evaluate()
        ee_quat: current EEF quaternion (xyzw) for rotation-lock reference
        """
        sem: List[Tuple] = []
        col: List[Tuple] = []
        pose_constrained = False
        desired_orientation = None
        v_limit: Optional[float] = None
        omega_limit: Optional[float] = None
        tilt_max: Optional[float] = None
        rule_names = []

        for ac in active_constraints:
            rule = ac.rule
            rule_names.append(rule.name)
            ctype = rule.cbf_type
            params = rule.cbf_params
            action = rule.violation_action

            if ctype == "spatial" and ac.obj_pos is not None:
                sq, alpha = self._build_spatial_sq(ac, params, action)
                sem.append((sq, alpha))

            elif ctype == "velocity":
                vlim = params.get("v_max", 0.15)
                v_limit = min(v_limit, vlim) if v_limit is not None else vlim

            elif ctype == "rotation":
                pose_constrained = True
                if desired_orientation is None:
                    desired_orientation = ee_quat.copy()
                # tilt constraint
                tl = params.get("tilt_max_deg", 15.0)
                tilt_max = math.radians(tl)
                al = params.get("omega_max", 0.25)
                omega_limit = min(omega_limit, al) if omega_limit is not None else al

            elif ctype == "boolean":
                # Block specific axes — encoded as tight workspace envelope
                pass  # handled upstream in filter.py

            elif ctype == "precondition":
                # Sequencing constraint — block is handled upstream
                pass

            # Any fragile / cautious object → add slow collision envelope
            if ac.obj_pos is not None:
                bbox = ac.obj_bbox if ac.obj_bbox is not None else np.array([0.04, 0.04, 0.04])
                sq = _box_sq(ac.obj_pos, bbox, margin=self.SAFETY_MARGIN)
                alpha = self._alpha_for_action(action)
                col.append((sq, alpha))

        if rule_names:
            logger.debug(f"[CBFMapper] active={rule_names} "
                         f"sem={len(sem)} col={len(col)} v_lim={v_limit}")

        return MappedCBFSet(
            semantic_envelopes=sem,
            collision_envelopes=col,
            pose_constrained=pose_constrained,
            desired_orientation=desired_orientation,
            velocity_limit=v_limit,
            angular_limit=omega_limit,
            tilt_max_rad=tilt_max,
            active_rule_names=rule_names,
        )

    def _build_spatial_sq(self, ac, params: Dict, action: str) -> Tuple:
        """Build a superquadric from the ActiveConstraint."""
        pos = ac.obj_pos
        bbox = ac.obj_bbox if ac.obj_bbox is not None else np.array([0.05, 0.05, 0.05])
        margin = params.get("margin", self.SPATIAL_MARGIN)
        shape = params.get("shape", "sphere")
        alpha = self._alpha_for_action(action, params)

        if shape == "sphere":
            radius = float(np.linalg.norm(bbox)) * 0.7 + margin
            sq = _sphere_sq(pos, max(radius, 0.06))
        elif shape == "half_space":
            # "above" relation: build vertical column
            sq = _above_sq(pos, bbox, self.ws_z_max)
        else:
            sq = _box_sq(pos, bbox, margin=margin)

        return sq, alpha

    @staticmethod
    def _alpha_for_action(action: str, params: Dict = None) -> float:
        """Map violation_action to CBF gain alpha."""
        params = params or {}
        scale = params.get("alpha_scale", 1.0)
        base = {
            "block": 2.0,
            "avoid": 1.0,
            "slow":  0.5,
            "ask":   0.25,
        }.get(action, 1.0)
        return base * scale
