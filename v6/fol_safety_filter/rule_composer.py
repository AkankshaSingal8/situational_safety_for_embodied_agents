"""
Rule Composer v5 — multi-obstacle, per-obstacle VLM-suggested radii.

Changes vs v4:
- Supports multiple obstacles from predicate_dict["obstacle_obs_keys"]
- Uses VLM-suggested avoidance radius per obstacle
- IS_FRAGILE → SLOW (velocity limit) not wider radius (avoids over-constraining)
- IS_SHARP → spatial avoidance
- Backward compat: still works with v4 single-obstacle predicate_dict
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compose_rules(predicate_dict: Dict, obstacle_obs_key: Optional[str] = None) -> List[Any]:
    from .kb import FOLRule, parse_formula

    rules: List[FOLRule] = []

    # Support both v4 (single obstacle) and v5 (multi-obstacle)
    obstacles = list(predicate_dict.get("obstacle_obs_keys") or [])
    if not obstacles and obstacle_obs_key:
        obstacles = [obstacle_obs_key]
    if not obstacles and predicate_dict.get("obstacle_obs_key"):
        obstacles = [predicate_dict["obstacle_obs_key"]]

    obstacle_radii = predicate_dict.get("obstacle_radii", {})
    obstacle_properties = predicate_dict.get("obstacle_properties", {})

    # backward compat: v4 single obstacle with physical_properties list
    if not obstacle_properties and obstacles:
        single_props = predicate_dict.get("physical_properties", [])
        obstacle_properties = {obs: single_props for obs in obstacles}

    if not obstacles:
        logger.warning("[RuleComposer-v5] No obstacles identified — no rules composed")
        return rules

    for obstacle in obstacles:
        props = obstacle_properties.get(obstacle, [])
        warning_r = obstacle_radii.get(obstacle, 0.18)

        # 1. Base spatial avoidance (always)
        formula = f"NEAR(eef, {obstacle}, {warning_r:.2f})"
        try:
            rules.append(FOLRule(
                name=f"OBSTACLE_AVOID_{obstacle.upper()}",
                formula=formula,
                parsed=parse_formula(formula),
                bindings={},
                cbf_type="spatial",
                cbf_params={"shape": "sphere", "margin": max(warning_r * 0.45, 0.05), "alpha_scale": 1.5},
                violation_action="avoid",
                primary_object=obstacle,
                description=f"VLM obstacle: {obstacle} (r={warning_r:.2f}m)",
                level=1,
            ))
            logger.info(f"[RuleComposer-v5] NEAR rule: {formula}")
        except Exception as e:
            logger.warning(f"[RuleComposer-v5] Base rule error for {obstacle}: {e}")

        # 2. IS_SPILLABLE → rotation lock
        if "IS_SPILLABLE" in props:
            r_formula = f"IS_SPILLABLE({obstacle})"
            try:
                rules.append(FOLRule(
                    name=f"SPILLABLE_LOCK_{obstacle.upper()}",
                    formula=r_formula,
                    parsed=parse_formula(r_formula),
                    bindings={},
                    cbf_type="rotation",
                    cbf_params={"angular_limit": 0.1},
                    violation_action="slow",
                    primary_object=obstacle,
                    description=f"{obstacle} is spillable — restrict wrist rotation",
                    level=2,
                ))
                logger.info(f"[RuleComposer-v5] Spillable rotation-lock: {r_formula}")
            except Exception as e:
                logger.debug(f"[RuleComposer-v5] Spillable rule error: {e}")

        # 3. IS_FRAGILE → velocity slow only (NOT wider radius)
        if "IS_FRAGILE" in props:
            capped_r = min(warning_r, 0.15)
            f_formula = f"NEAR(eef, {obstacle}, {capped_r:.2f})"
            try:
                rules.append(FOLRule(
                    name=f"FRAGILE_SLOW_{obstacle.upper()}",
                    formula=f_formula,
                    parsed=parse_formula(f_formula),
                    bindings={},
                    cbf_type="velocity",
                    cbf_params={"v_max": 0.10},
                    violation_action="slow",
                    primary_object=obstacle,
                    description=f"{obstacle} is fragile — slow approach within {capped_r:.2f}m",
                    level=2,
                ))
                logger.info(f"[RuleComposer-v5] Fragile slow rule: {f_formula}")
            except Exception as e:
                logger.debug(f"[RuleComposer-v5] Fragile rule error: {e}")

        # 4. IS_HOT → larger hard exclusion zone
        if "IS_HOT" in props:
            hot_r = min(warning_r, 0.25)
            h_formula = f"NEAR(eef, {obstacle}, {hot_r:.2f})"
            try:
                rules.append(FOLRule(
                    name=f"HOT_ZONE_{obstacle.upper()}",
                    formula=h_formula,
                    parsed=parse_formula(h_formula),
                    bindings={},
                    cbf_type="spatial",
                    cbf_params={"shape": "sphere", "margin": 0.10, "alpha_scale": 2.0},
                    violation_action="avoid",
                    primary_object=obstacle,
                    description=f"{obstacle} is hot — {hot_r:.2f}m exclusion zone",
                    level=2,
                ))
                logger.info(f"[RuleComposer-v5] Hot-zone rule: {h_formula}")
            except Exception as e:
                logger.debug(f"[RuleComposer-v5] Hot rule error: {e}")

        # 5. IS_SHARP → spatial avoidance
        if "IS_SHARP" in props:
            sharp_r = min(warning_r, 0.15)
            s_formula = f"NEAR(eef, {obstacle}, {sharp_r:.2f})"
            try:
                rules.append(FOLRule(
                    name=f"SHARP_AVOID_{obstacle.upper()}",
                    formula=s_formula,
                    parsed=parse_formula(s_formula),
                    bindings={},
                    cbf_type="spatial",
                    cbf_params={"shape": "sphere", "margin": 0.08, "alpha_scale": 1.5},
                    violation_action="avoid",
                    primary_object=obstacle,
                    description=f"{obstacle} is sharp — avoid within {sharp_r:.2f}m",
                    level=2,
                ))
                logger.info(f"[RuleComposer-v5] Sharp rule: {s_formula}")
            except Exception as e:
                logger.debug(f"[RuleComposer-v5] Sharp rule error: {e}")

    return rules
