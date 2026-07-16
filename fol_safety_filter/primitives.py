"""
FOL Safety Filter — Level 1 Primitive Predicate Vocabulary

All predicates follow the signature:
    predicate(state: RobotState, **kwargs) -> bool

RobotState is a lightweight dataclass populated at each timestep from
MuJoCo obs (sim) or real sensors (hardware).

Grounding strategy:
  - Spatial / robot-state predicates: purely geometric, <1 ms, deterministic
  - Physical-state predicates: VLM binary query, cached once per episode
  - The cache lives in FOLKnowledgeBase (kb.py); predicates here are stateless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# State container (passed to every predicate at runtime)
# ---------------------------------------------------------------------------

@dataclass
class ObjectState:
    name: str
    pos: np.ndarray           # (3,) world frame
    quat: np.ndarray          # (4,) xyzw
    bbox_half: np.ndarray     # (3,) half-extents in object frame
    # populated by VLM at episode start (None = not yet queried)
    is_fragile: Optional[bool] = None
    is_wet: Optional[bool] = None
    is_lit: Optional[bool] = None
    is_full: Optional[bool] = None
    is_open: Optional[bool] = None
    is_flammable: Optional[bool] = None
    is_absorbent: Optional[bool] = None
    is_human: Optional[bool] = None
    has_sharp_part: Optional[bool] = None
    is_heavy: Optional[bool] = None
    is_spillable: Optional[bool] = None  # derived: is_full AND is_open


@dataclass
class RobotState:
    ee_pos: np.ndarray              # (3,) end-effector position, world frame
    ee_quat: np.ndarray             # (4,) end-effector orientation xyzw
    ee_vel: np.ndarray              # (3,) end-effector linear velocity
    joint_pos: np.ndarray           # (7,) joint angles
    joint_vel: np.ndarray           # (7,) joint velocities
    joint_limits_lo: np.ndarray     # (7,)
    joint_limits_hi: np.ndarray     # (7,)
    gripper_width: float            # metres, 0=closed 1=open
    objects: Dict[str, ObjectState] = field(default_factory=dict)
    # workspace AABB
    ws_lo: np.ndarray = field(default_factory=lambda: np.array([-0.5, -0.3, 0.81]))
    ws_hi: np.ndarray = field(default_factory=lambda: np.array([ 0.5,  0.6, 1.40]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _dist_xy(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _tilt_angle(quat: np.ndarray) -> float:
    """Angle (radians) between the object's local z-axis and world z-axis."""
    x, y, z, w = quat
    # rotate world z=[0,0,1] by quat
    local_z = np.array([
        2*(x*z + w*y),
        2*(y*z - w*x),
        1 - 2*(x*x + y*y),
    ])
    cos_a = float(np.clip(local_z[2], -1.0, 1.0))
    return math.acos(cos_a)


# ---------------------------------------------------------------------------
# LEVEL 1a — Spatial predicates (geometric, deterministic)
# ---------------------------------------------------------------------------

def NEAR(state: RobotState, obj_a: str, obj_b: str, threshold: float = 0.15) -> bool:
    """dist(a, b) < threshold metres."""
    a = state.objects[obj_a].pos if obj_a != "eef" else state.ee_pos
    b = state.objects[obj_b].pos if obj_b != "eef" else state.ee_pos
    return _dist(a, b) < threshold


def ABOVE(state: RobotState, obj_a: str, obj_b: str, margin: float = 0.02) -> bool:
    """Centre of a is strictly above centre of b (z-axis)."""
    za = state.objects[obj_a].pos[2] if obj_a != "eef" else state.ee_pos[2]
    zb = state.objects[obj_b].pos[2] if obj_b != "eef" else state.ee_pos[2]
    return za > zb + margin


def BELOW(state: RobotState, obj_a: str, obj_b: str, margin: float = 0.02) -> bool:
    """Centre of a is strictly below centre of b."""
    return ABOVE(state, obj_b, obj_a, margin)


def SUPPORTS(state: RobotState, obj_a: str, obj_b: str,
             xy_thresh: float = 0.06, z_margin: float = 0.01) -> bool:
    """a physically supports b: b is above a AND their xy centres are close."""
    return ABOVE(state, obj_b, obj_a, z_margin) and \
           _dist_xy(state.objects[obj_a].pos, state.objects[obj_b].pos) < xy_thresh


def IN_WORKSPACE(state: RobotState, margin: float = 0.02) -> bool:
    """EEF is within the workspace AABB (with margin)."""
    lo = state.ws_lo + margin
    hi = state.ws_hi - margin
    return bool(np.all(state.ee_pos >= lo) and np.all(state.ee_pos <= hi))


def NEAR_EDGE(state: RobotState, edge_margin: float = 0.05) -> bool:
    """EEF is within edge_margin of any workspace boundary."""
    lo_dist = state.ee_pos - state.ws_lo
    hi_dist = state.ws_hi - state.ee_pos
    return bool(np.any(lo_dist < edge_margin) or np.any(hi_dist < edge_margin))


def NEAR_JOINT_LIMIT(state: RobotState, joint_idx: int,
                     margin_rad: float = 0.15) -> bool:
    """Joint j is within margin_rad of its hard limit."""
    q = state.joint_pos[joint_idx]
    lo = state.joint_limits_lo[joint_idx]
    hi = state.joint_limits_hi[joint_idx]
    return (q - lo) < margin_rad or (hi - q) < margin_rad


def ANY_JOINT_NEAR_LIMIT(state: RobotState, margin_rad: float = 0.15) -> bool:
    """Any joint is close to its limit."""
    return any(NEAR_JOINT_LIMIT(state, j, margin_rad)
               for j in range(len(state.joint_pos)))


def TILTED(state: RobotState, obj: str, threshold_deg: float = 20.0) -> bool:
    """Object tilt from upright exceeds threshold."""
    quat = state.objects[obj].quat
    angle_rad = _tilt_angle(quat)
    return math.degrees(angle_rad) > threshold_deg


def EEF_TILTED(state: RobotState, threshold_deg: float = 20.0) -> bool:
    """End-effector tilt exceeds threshold (relevant when holding spillable)."""
    return math.degrees(_tilt_angle(state.ee_quat)) > threshold_deg


# ---------------------------------------------------------------------------
# LEVEL 1b — Robot-state predicates (sensor-derived)
# ---------------------------------------------------------------------------

def MOVING_FAST(state: RobotState, v_thresh: float = 0.25) -> bool:
    """EEF translational speed exceeds threshold (m/s)."""
    return float(np.linalg.norm(state.ee_vel)) > v_thresh


def GRIPPING(state: RobotState, obj: str, proximity: float = 0.08) -> bool:
    """Gripper is at least half-closed AND near the object."""
    return state.gripper_width < 0.5 and NEAR(state, "eef", obj, proximity)


def HOLDING(state: RobotState, obj: str) -> bool:
    """Object is being actively held (gripped + lifted off table)."""
    obj_z = state.objects[obj].pos[2]
    table_z = state.ws_lo[2]
    lifted = obj_z > table_z + 0.02
    return GRIPPING(state, obj) and lifted


# ---------------------------------------------------------------------------
# LEVEL 1b — Physical-state predicates (VLM-cached, Boolean getters)
# Each returns the cached value from ObjectState; grounding happens in
# VLMEpisodeGrounder (vlm_grounder.py) before the episode starts.
# ---------------------------------------------------------------------------

def _cached(state: RobotState, obj: str, attr: str) -> Optional[bool]:
    o = state.objects.get(obj)
    if o is None:
        return None
    return getattr(o, attr, None)


def IS_FRAGILE(state: RobotState, obj: str) -> bool:
    return bool(_cached(state, obj, "is_fragile"))


def IS_WET(state: RobotState, obj: str) -> bool:
    return bool(_cached(state, obj, "is_wet"))


def IS_LIT(state: RobotState, obj: str) -> bool:
    return bool(_cached(state, obj, "is_lit"))


def IS_FULL(state: RobotState, obj: str) -> bool:
    return bool(_cached(state, obj, "is_full"))


def IS_OPEN(state: RobotState, obj: str) -> bool:
    return bool(_cached(state, obj, "is_open"))


def IS_FLAMMABLE(state: RobotState, obj: str) -> bool:
    return bool(_cached(state, obj, "is_flammable"))


def IS_ABSORBENT(state: RobotState, obj: str) -> bool:
    return bool(_cached(state, obj, "is_absorbent"))


def IS_HUMAN(state: RobotState, obj: str) -> bool:
    return bool(_cached(state, obj, "is_human"))


def HAS_SHARP_PART(state: RobotState, obj: str) -> bool:
    return bool(_cached(state, obj, "has_sharp_part"))


def IS_HEAVY(state: RobotState, obj: str) -> bool:
    return bool(_cached(state, obj, "is_heavy"))


def IS_SPILLABLE(state: RobotState, obj: str) -> bool:
    """Derived: container is spillable if it is full AND open."""
    full = _cached(state, obj, "is_full")
    open_ = _cached(state, obj, "is_open")
    if full is None or open_ is None:
        return False
    return bool(full and open_)


# ---------------------------------------------------------------------------
# Predicate registry
# Maps string name → callable(state, *args) -> bool
# Used by the KB formula evaluator.
# ---------------------------------------------------------------------------

PREDICATE_REGISTRY: Dict[str, object] = {
    # spatial
    "NEAR":              NEAR,
    "ABOVE":             ABOVE,
    "BELOW":             BELOW,
    "SUPPORTS":          SUPPORTS,
    "IN_WORKSPACE":      IN_WORKSPACE,
    "NEAR_EDGE":         NEAR_EDGE,
    "NEAR_JOINT_LIMIT":  NEAR_JOINT_LIMIT,
    "ANY_JOINT_NEAR_LIMIT": ANY_JOINT_NEAR_LIMIT,
    "TILTED":            TILTED,
    "EEF_TILTED":        EEF_TILTED,
    # robot-state
    "MOVING_FAST":       MOVING_FAST,
    "GRIPPING":          GRIPPING,
    "HOLDING":           HOLDING,
    # physical-state (VLM-cached)
    "IS_FRAGILE":        IS_FRAGILE,
    "IS_WET":            IS_WET,
    "IS_LIT":            IS_LIT,
    "IS_FULL":           IS_FULL,
    "IS_OPEN":           IS_OPEN,
    "IS_FLAMMABLE":      IS_FLAMMABLE,
    "IS_ABSORBENT":      IS_ABSORBENT,
    "IS_HUMAN":          IS_HUMAN,
    "HAS_SHARP_PART":    HAS_SHARP_PART,
    "IS_HEAVY":          IS_HEAVY,
    "IS_SPILLABLE":      IS_SPILLABLE,
}


# ---------------------------------------------------------------------------
# VLM grounding queries for physical-state predicates
# Used by VLMEpisodeGrounder to build the binary-question prompts.
# ---------------------------------------------------------------------------

VLM_GROUNDING_QUERIES: Dict[str, str] = {
    "is_fragile":    "Is the {obj} fragile or easily breakable if dropped or squeezed?",
    "is_wet":        "Is the {obj} wet, damp, or does it contain liquid?",
    "is_lit":        "Is the {obj} on fire, lit, or actively producing heat or flame?",
    "is_full":       "Is the {obj} full or mostly filled with contents?",
    "is_open":       "Is the {obj} open or uncovered such that contents could spill?",
    "is_flammable":  "Is the {obj} flammable or combustible (e.g. paper, fabric, wood, alcohol)?",
    "is_absorbent":  "Is the {obj} water-absorbent (e.g. paper, fabric, sponge, wood)?",
    "is_human":      "Is the {obj} a human body part (hand, arm, face, etc.)?",
    "has_sharp_part":"Does the {obj} have a sharp, pointed, or cutting region that is dangerous to touch?",
    "is_heavy":      "Is the {obj} heavy (over 2 kg), requiring careful handling?",
}


# ---------------------------------------------------------------------------
# CBF parameter templates for each predicate → used by cbf_mapper.py
# ---------------------------------------------------------------------------

@dataclass
class CBFTemplate:
    cbf_type: str          # "spatial" | "velocity" | "rotation" | "boolean" | "precondition"
    violation_action: str  # "block" | "slow" | "avoid" | "ask"
    params: Dict           # type-specific parameters


PREDICATE_CBF_TEMPLATES: Dict[str, CBFTemplate] = {
    "NEAR": CBFTemplate(
        cbf_type="spatial",
        violation_action="avoid",
        params={"shape": "sphere", "alpha_scale": 1.0},
    ),
    "ABOVE": CBFTemplate(
        cbf_type="spatial",
        violation_action="avoid",
        params={"shape": "half_space", "axis": "z", "alpha_scale": 1.0},
    ),
    "BELOW": CBFTemplate(
        cbf_type="spatial",
        violation_action="avoid",
        params={"shape": "half_space", "axis": "-z", "alpha_scale": 1.0},
    ),
    "NEAR_EDGE": CBFTemplate(
        cbf_type="boolean",
        violation_action="block",
        params={"block_axes": ["x", "y"]},
    ),
    "ANY_JOINT_NEAR_LIMIT": CBFTemplate(
        cbf_type="boolean",
        violation_action="block",
        params={},
    ),
    "MOVING_FAST": CBFTemplate(
        cbf_type="velocity",
        violation_action="slow",
        params={"v_max": 0.20},
    ),
    "IS_FRAGILE": CBFTemplate(
        cbf_type="velocity",
        violation_action="slow",
        params={"v_max": 0.10, "alpha_scale": 0.25},
    ),
    "IS_HUMAN": CBFTemplate(
        cbf_type="velocity",
        violation_action="slow",
        params={"v_max": 0.10, "spatial_margin": 0.50},
    ),
    "IS_SPILLABLE": CBFTemplate(
        cbf_type="rotation",
        violation_action="block",
        params={"omega_max": 0.30, "tilt_max_deg": 15.0},
    ),
    "HAS_SHARP_PART": CBFTemplate(
        cbf_type="spatial",
        violation_action="avoid",
        params={"shape": "sphere", "margin": 0.10, "alpha_scale": 0.5},
    ),
    "IS_LIT": CBFTemplate(
        cbf_type="spatial",
        violation_action="avoid",
        params={"shape": "sphere", "margin": 0.30, "alpha_scale": 1.0},
    ),
    "EEF_TILTED": CBFTemplate(
        cbf_type="rotation",
        violation_action="block",
        params={"tilt_max_deg": 20.0},
    ),
    "SUPPORTS": CBFTemplate(
        cbf_type="precondition",
        violation_action="block",
        params={"message": "Object is supporting another — clear the top object first."},
    ),
}


# ---------------------------------------------------------------------------
# Composed rule templates (Level 2 seeds)
# The VLM can extend/override these at runtime.
# ---------------------------------------------------------------------------

COMPOSED_RULE_SEEDS: List[Dict] = [
    {
        "name": "FIRE_HAZARD",
        "formula": "IS_LIT(A) AND IS_FLAMMABLE(B) AND NEAR(eef, A, 0.30)",
        "cbf_type": "spatial",
        "cbf_params": {"shape": "sphere", "margin": 0.30, "alpha_scale": 1.0},
        "violation_action": "avoid",
        "description": "EEF near a lit object when a flammable object is present.",
    },
    {
        "name": "WATER_DAMAGE",
        "formula": "IS_WET(A) AND IS_ABSORBENT(B) AND ABOVE(A, B) AND HOLDING(eef, A)",
        "cbf_type": "spatial",
        "cbf_params": {"shape": "half_space", "axis": "z", "alpha_scale": 0.5},
        "violation_action": "avoid",
        "description": "Holding a wet object above an absorbent object.",
    },
    {
        "name": "SPILL_RISK",
        "formula": "IS_SPILLABLE(A) AND HOLDING(eef, A)",
        "cbf_type": "rotation",
        "cbf_params": {"omega_max": 0.20, "tilt_max_deg": 12.0},
        "violation_action": "block",
        "description": "Holding a full, open container — lock rotation.",
    },
    {
        "name": "FRAGILE_GRASP",
        "formula": "IS_FRAGILE(A) AND GRIPPING(eef, A)",
        "cbf_type": "velocity",
        "cbf_params": {"v_max": 0.08, "alpha_scale": 0.25},
        "violation_action": "slow",
        "description": "Grasping a fragile object — reduce speed and force.",
    },
    {
        "name": "TOPPLE_RISK",
        "formula": "SUPPORTS(A, B) AND NEAR(eef, A, 0.20)",
        "cbf_type": "velocity",
        "cbf_params": {"v_max": 0.10},
        "violation_action": "slow",
        "description": "Approaching an object that supports a stack.",
    },
    {
        "name": "HUMAN_PROXIMITY",
        "formula": "IS_HUMAN(A) AND NEAR(eef, A, 0.50)",
        "cbf_type": "velocity",
        "cbf_params": {"v_max": 0.10, "spatial_margin": 0.50},
        "violation_action": "slow",
        "description": "EEF within 50 cm of a human body part.",
    },
    {
        "name": "SHARP_APPROACH",
        "formula": "HAS_SHARP_PART(A) AND NEAR(eef, A, 0.15)",
        "cbf_type": "spatial",
        "cbf_params": {"shape": "sphere", "margin": 0.10, "alpha_scale": 0.5},
        "violation_action": "avoid",
        "description": "Approaching the sharp region of an object.",
    },
]
