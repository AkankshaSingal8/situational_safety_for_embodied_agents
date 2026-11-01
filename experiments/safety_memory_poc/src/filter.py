"""Keep-away action filter (discrete control-barrier projection).

The enforcement mechanism is deliberately the same in every arm. Only the set of
active constraints differs, and that set is produced by the belief source. So any
behavioural difference between arms is attributable to what was believed, not to
how it was enforced.

Barrier, per active constraint:      h(p) = ||p - o|| - margin
Discrete decrease condition:         h(p_next) >= (1 - gamma) * h(p)
Linearised in the commanded delta u: n . (scale * u) >= -gamma * h,  n = (p-o)/||p-o||

When the condition is violated the action is repaired by the minimum-norm
addition along n, which is the closed-form projection onto that half-space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class FilterReport:
    active: int = 0
    intervened: bool = False
    deviation: float = 0.0
    min_h: float = float("inf")
    details: List[dict] = None

    def as_dict(self) -> dict:
        return {"active": self.active, "intervened": self.intervened,
                "deviation": float(self.deviation),
                "min_h": None if self.min_h == float("inf") else float(self.min_h)}


def apply_keep_away_multi(u: np.ndarray,
                          points: Dict[str, np.ndarray],
                          point_scales: Dict[str, float],
                          constraints: Sequence,
                          target_pos: Dict[str, np.ndarray],
                          action_scale: float = 0.0113,
                          gamma: float = 0.5,
                          u_max: float = 1.0,
                          sweeps: int = 3,
                          goal: np.ndarray = None,
                          slide: float = 0.0,
                          lead: str = None) -> Tuple[np.ndarray, FilterReport]:
    """Enforce every constraint at every robot checkpoint.

    Each checkpoint carries a margin scale, so a wrist or elbow can be given a
    smaller standoff than the gripper without changing the rule. `lead` names the
    checkpoint used for the tangential slide reference.
    """
    u_out = np.array(u, dtype=float).copy()
    rep = FilterReport(active=len(constraints) * max(1, len(points)), details=[])
    if not constraints or not points:
        rep.active = 0
        return u_out, rep
    u0 = u_out[:3].copy()

    for _ in range(max(1, sweeps)):
        changed = False
        for pname, p in points.items():
            scale = point_scales.get(pname, 1.0)
            for c in constraints:
                o = np.asarray(target_pos[c.target], dtype=float)
                d = np.asarray(p, dtype=float) - o
                dist = float(np.linalg.norm(d))
                margin = c.margin * scale
                if dist < 1e-9:
                    n = np.array([0.0, 0.0, 1.0]); h = -margin
                else:
                    n = d / dist; h = dist - margin
                rep.min_h = min(rep.min_h, h)
                lhs = float(np.dot(n, action_scale * u_out[:3]))
                rhs = -gamma * h
                if lhs < rhs - 1e-12:
                    lam = (rhs - lhs) / action_scale
                    u_out[:3] = u_out[:3] + lam * n
                    changed = True
                    rep.details.append({"point": pname, "target": c.target,
                                        "h": h, "lam": float(lam),
                                        "provenance": c.provenance})
        if not changed:
            break

    if slide > 0.0 and goal is not None and rep.details:
        ref = points.get(lead) if lead else None
        if ref is None:
            ref = next(iter(points.values()))
        c = constraints[0]
        o = np.asarray(target_pos[c.target], dtype=float)
        d = np.asarray(ref, dtype=float) - o
        nrm = float(np.linalg.norm(d))
        if nrm > 1e-9:
            n = d / nrm
            g = np.asarray(goal, dtype=float) - np.asarray(ref, dtype=float)
            gn = float(np.linalg.norm(g))
            if gn > 1e-9:
                g = g / gn
                tang = g - np.dot(g, n) * n
                tn = float(np.linalg.norm(tang))
                if tn <= 1e-6:
                    up = np.array([0.0, 0.0, 1.0])
                    tang = up - np.dot(up, n) * n
                    tn = float(np.linalg.norm(tang))
                if tn > 1e-6:
                    u_out[:3] = u_out[:3] + slide * (tang / tn)

    u_out[:3] = np.clip(u_out[:3], -u_max, u_max)
    rep.deviation = float(np.linalg.norm(u_out[:3] - u0))
    rep.intervened = rep.deviation > 1e-9
    return u_out, rep


def apply_keep_away(u: np.ndarray,
                    eef_pos: np.ndarray,
                    constraints: Sequence,
                    target_pos: Dict[str, np.ndarray],
                    action_scale: float = 0.05,
                    gamma: float = 0.5,
                    u_max: float = 1.0,
                    sweeps: int = 3,
                    goal: np.ndarray = None,
                    slide: float = 0.0) -> Tuple[np.ndarray, FilterReport]:
    """Repair the translational part of `u` so every constraint's barrier holds.

    `u` is the nominal action; only its first three (translational) components
    are touched. Returns the certified action and a report.
    """
    u_out = np.array(u, dtype=float).copy()
    rep = FilterReport(active=len(constraints), details=[])
    if not constraints:
        return u_out, rep

    u0 = u_out[:3].copy()
    for _ in range(max(1, sweeps)):
        changed = False
        for c in constraints:
            o = np.asarray(target_pos[c.target], dtype=float)
            d = np.asarray(eef_pos, dtype=float) - o
            dist = float(np.linalg.norm(d))
            if dist < 1e-9:
                # Degenerate coincidence: push along +z rather than divide by zero.
                n = np.array([0.0, 0.0, 1.0])
                h = -c.margin
            else:
                n = d / dist
                h = dist - c.margin
            rep.min_h = min(rep.min_h, h)
            lhs = float(np.dot(n, action_scale * u_out[:3]))
            rhs = -gamma * h
            if lhs < rhs - 1e-12:
                # Minimum-norm correction along n to make the constraint tight.
                lam = (rhs - lhs) / action_scale
                u_out[:3] = u_out[:3] + lam * n
                changed = True
                rep.details.append({"target": c.target, "h": h, "lam": float(lam),
                                    "provenance": c.provenance})
        if not changed:
            break

    # Tangential slide. A min-norm projection preserves tangential motion, but a
    # proportional controller aiming straight through the protected object has
    # almost no tangential component, so the repaired action collapses to nearly
    # zero and the arm stalls in front of the obstacle. Adding an explicit
    # tangential term lets it travel around. This is a property of the
    # ENFORCEMENT layer and is applied identically in every arm, so it cannot
    # bias the memory comparison; it only removes a controller artefact.
    if slide > 0.0 and goal is not None and rep.details:
        c = constraints[0]
        o = np.asarray(target_pos[c.target], dtype=float)
        d = np.asarray(eef_pos, dtype=float) - o
        nrm = float(np.linalg.norm(d))
        if nrm > 1e-9:
            n = d / nrm
            g = np.asarray(goal, dtype=float) - np.asarray(eef_pos, dtype=float)
            gn = float(np.linalg.norm(g))
            if gn > 1e-9:
                g = g / gn
                tang = g - np.dot(g, n) * n
                tn = float(np.linalg.norm(tang))
                if tn <= 1e-6:
                    # Exactly head-on: the goal is diametrically opposite the
                    # obstacle and no tangential direction is defined. Prefer
                    # lifting over it, which is the natural tabletop detour;
                    # fall back to any orthogonal direction if n is vertical.
                    up = np.array([0.0, 0.0, 1.0])
                    tang = up - np.dot(up, n) * n
                    tn = float(np.linalg.norm(tang))
                    if tn <= 1e-6:
                        alt = np.array([1.0, 0.0, 0.0])
                        tang = alt - np.dot(alt, n) * n
                        tn = float(np.linalg.norm(tang))
                if tn > 1e-6:
                    u_out[:3] = u_out[:3] + slide * (tang / tn)

    u_out[:3] = np.clip(u_out[:3], -u_max, u_max)
    rep.deviation = float(np.linalg.norm(u_out[:3] - u0))
    rep.intervened = rep.deviation > 1e-9
    return u_out, rep
