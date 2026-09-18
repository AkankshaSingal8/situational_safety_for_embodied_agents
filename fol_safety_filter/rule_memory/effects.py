"""
FOL Safety Filter — effect handlers: one per effect predicate.

The filter used to decide geometry itself, in a loop over
``self._obstacle_names`` with literal radii. These handlers replace that loop:
each reads its numbers from the rule record that fired and applies exactly one
kind of correction. Deleting a record removes exactly one behaviour because
nothing here supplies a default for a rule that is absent.

Ordering is semantics, not style. ``_apply_point_cbf`` mutates the action in
place, so the sequence of corrections changes the result. ``order_key``
reproduces the obstacle-major nesting of the loop this module replaces: for
each obstacle, the end-effector projection, then each arm checkpoint in body
order. ``tools/fol_parity_trace.py`` holds that to ``max|delta| == 0``.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Effects that project the translational action and therefore care about order.
SPATIAL_PREDICATES = ("avoid_sphere", "avoid_region_above")

# Effects that clamp a magnitude; strictest wins, so order is irrelevant.
LIMIT_PREDICATES = ("limit_speed", "limit_angular_rate")


def order_key(rule) -> Tuple[int, int, int]:
    """Sort key reproducing the pre-decomposition loop's nesting."""
    return (
        getattr(rule, "target_index", 0),
        getattr(rule, "subject_rank", 0),
        getattr(rule, "checkpoint_index", 0),
    )


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --------------------------------------------------------------------------- #
# avoid_sphere
# --------------------------------------------------------------------------- #

def resolve_avoid_sphere(
    rule,
    obs_name: str,
    subject_binding: str,
    obstacle_radii: Dict[str, float],
    obstacle_ellipsoids: Dict[str, Dict],
    checkpoint: Optional[Dict],
    aiming_at_target: bool,
) -> Dict:
    """Turn a fired ``avoid_sphere`` rule into `_apply_point_cbf` arguments.

    Three pieces of precedence, all of which the parity trace pins:

    * ``warning_r`` from the record is a **floor**, not a radius. It replaces
      ``max(0.20, self._obstacle_radii.get(obs_name, 0.20))``, so a grounded
      per-obstacle radius may widen the zone and may never narrow it. Lowering
      the record below what perception asked for has no effect, deliberately.
    * A checkpoint dict may carry its own radii, and then they win. Only the
      opt-in ``FOL_HAND_CHECKPOINT`` bodies do; the default elbow and wrist
      carry position and name alone, so their radii come from the record and
      editing the record changes what the arm does.
    * Ellipsoid zones are used only when the record asks for them *and*
      ``FOL_ELLIPSOID`` is on *and* the grounder actually produced an oriented
      extent. ``FOL_RAY_RESCALE=target`` additionally needs the command to be
      aiming at a detected target.
    """
    params = rule.cbf_params
    if checkpoint is not None and "warning_r" in checkpoint:
        warning_r = float(checkpoint["warning_r"])
        hard_r = float(checkpoint["hard_r"])
        push_hard = float(checkpoint["push"])
    else:
        warning_r = float(params["warning_r"])
        hard_r = float(params["hard_r"])
        push_hard = float(params["push_hard"])

    if subject_binding == "eef":
        warning_r = max(warning_r, float(obstacle_radii.get(obs_name, warning_r)))

    ellipsoid = None
    if (params.get("shape") == "ellipsoid_if_available"
            and _env("FOL_ELLIPSOID", "1") == "1"):
        ellipsoid = obstacle_ellipsoids.get(obs_name)

    cancel_scale = 1.0
    if aiming_at_target:
        cancel_scale = float(params.get("cancel_scale_on_target_corridor", 1.0))

    rr_mode = _env("FOL_RAY_RESCALE", "0")
    ray_rescale = rr_mode == "1" or (rr_mode == "target" and aiming_at_target)

    return {
        "warning_r": warning_r,
        "hard_r": hard_r,
        "push_hard": push_hard,
        "cancel_scale": cancel_scale,
        "ray_rescale": ray_rescale,
        "frame_R": ellipsoid["frame_R"] if ellipsoid else None,
        "warn_radii": ellipsoid["warn_radii"] if ellipsoid else None,
        "hard_radii": ellipsoid["hard_radii"] if ellipsoid else None,
    }


# --------------------------------------------------------------------------- #
# avoid_region_above
# --------------------------------------------------------------------------- #

def apply_avoid_region_above(
    u: np.ndarray,
    subject_point: np.ndarray,
    target_pos: np.ndarray,
    target_bbox_half: np.ndarray,
    params: Dict,
) -> bool:
    """Keep the subject out of the column standing above the target, in place.

    The prohibited set is the target's horizontal footprint, expanded by
    ``clearance_m``, extruded from the target's top surface up to ``z_max``.
    This is a projected-region constraint, not proximity: a payload 40 cm
    directly above a bowl is outside every avoidance sphere here and squarely
    inside this region.

    Inside the column, the horizontal component of the commanded motion that
    would drive the subject further in is cancelled and replaced by an outward
    push along the shorter escape axis — so the correction routes the payload
    out sideways rather than lifting it, since lifting does not leave a column
    that extends to the ceiling.

    Returns whether it intervened, for the caller's metrics.
    """
    clearance = float(params.get("clearance_m", 0.02))
    z_max = float(params.get("z_max", 1.40))
    cancel_scale = float(params.get("cancel_scale", 1.0))

    half = np.maximum(np.asarray(target_bbox_half, dtype=np.float64)[:2], 0.03)
    half = half + clearance

    delta_xy = np.asarray(subject_point[:2], dtype=np.float64) - target_pos[:2]
    inside_xy = bool(np.all(np.abs(delta_xy) <= half))
    z_floor = float(target_pos[2])
    inside_z = z_floor < float(subject_point[2]) <= z_max
    if not (inside_xy and inside_z):
        return False

    # Escape along whichever horizontal axis needs the least travel; ties go
    # to x, deterministically, so the correction is reproducible.
    slack = half - np.abs(delta_xy)
    axis = 0 if slack[0] <= slack[1] else 1
    sign = 1.0 if delta_xy[axis] >= 0.0 else -1.0

    intervened = False
    inward = -sign * u[axis]
    if inward > 0.0:
        u[axis] += inward * cancel_scale * sign
        intervened = True

    depth = float(slack[axis])
    if depth > 0.0:
        u[axis] += sign * depth * cancel_scale
        intervened = True

    return intervened


# --------------------------------------------------------------------------- #
# limit_speed / limit_angular_rate
# --------------------------------------------------------------------------- #

def collect_speed_limit(rules: Sequence) -> Optional[float]:
    """Strictest ``v_max`` among the fired ``limit_speed`` rules."""
    values = [
        float(r.cbf_params["v_max"])
        for r in rules
        if r.cbf_type == "limit_speed" and "v_max" in r.cbf_params
    ]
    return min(values) if values else None


def collect_angular_limit(rules: Sequence) -> Optional[float]:
    """Strictest ``omega_max`` among the fired ``limit_angular_rate`` rules."""
    values = [
        float(r.cbf_params["omega_max"])
        for r in rules
        if r.cbf_type == "limit_angular_rate" and "omega_max" in r.cbf_params
    ]
    return min(values) if values else None
