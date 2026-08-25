"""
Rule Composer — v4

Converts VLMObstacleGrounder predicate_dict into List[FOLRule] objects
for the knowledge base. Composes rules at episode start based on:

  1. Base NEAR avoidance rule (always)
  2. IS_SPILLABLE → rotation lock (prevent wrist rotation when carrying liquid)
  3. IS_FRAGILE   → wider warning radius (0.30m vs default 0.20m)
  4. IS_HOT       → larger hard exclusion zone (0.25m vs default 0.10m)

The base avoidance geometry (v3 carrying-phase CBF) is unaffected — these
rules augment the semantic layer (FOLKnowledgeBase), not the geometric layer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compose_rules(predicate_dict: Dict, obstacle_obs_key: Optional[str]) -> List[Any]:
    """
    Build FOLRule list from VLM predicate output.

    predicate_dict keys:
      obstacle_obs_key: str | None
      physical_properties: List[str]   e.g. ["IS_SPILLABLE", "IS_FRAGILE"]

    Returns List[FOLRule] ready to pass to FOLKnowledgeBase.
    """
    # Import here to avoid circular import at module load
    from .kb import FOLRule, parse_formula

    rules: List[FOLRule] = []
    obstacle = obstacle_obs_key or predicate_dict.get("obstacle_obs_key")
    props: List[str] = predicate_dict.get("physical_properties", [])

    if not obstacle:
        logger.warning("[RuleComposer] No obstacle identified — no rules composed")
        return rules

    # 1. Base spatial avoidance (always)
    formula = f"NEAR(eef, {obstacle}, 0.20)"
    try:
        rules.append(FOLRule(
            name=f"OBSTACLE_AVOID_{obstacle.upper()}",
            formula=formula,
            parsed=parse_formula(formula),
            bindings={},
            cbf_type="spatial",
            cbf_params={"shape": "sphere", "margin": 0.08, "alpha_scale": 1.5},
            violation_action="avoid",
            primary_object=obstacle,
            description=f"VLM-identified obstacle: {obstacle}",
            level=1,
        ))
        logger.info(f"[RuleComposer] Base NEAR rule: {formula}")
    except Exception as e:
        logger.warning(f"[RuleComposer] Could not parse base rule: {e}")

    # 2. IS_SPILLABLE → rotation lock when holding
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
            logger.info(f"[RuleComposer] Spillable rotation-lock rule: {r_formula}")
        except Exception as e:
            logger.debug(f"[RuleComposer] Spillable rule error: {e}")

    # 3. IS_FRAGILE → wider warning radius (injected via cbf_params override)
    if "IS_FRAGILE" in props:
        f_formula = f"NEAR(eef, {obstacle}, 0.30)"
        try:
            rules.append(FOLRule(
                name=f"FRAGILE_WIDER_{obstacle.upper()}",
                formula=f_formula,
                parsed=parse_formula(f_formula),
                bindings={},
                cbf_type="spatial",
                cbf_params={"shape": "sphere", "margin": 0.12, "alpha_scale": 2.0},
                violation_action="avoid",
                primary_object=obstacle,
                description=f"{obstacle} is fragile — extra clearance at 0.30m",
                level=2,
            ))
            logger.info(f"[RuleComposer] Fragile wider-radius rule: {f_formula}")
        except Exception as e:
            logger.debug(f"[RuleComposer] Fragile rule error: {e}")

    # 4. IS_HOT → larger exclusion zone
    if "IS_HOT" in props:
        h_formula = f"NEAR(eef, {obstacle}, 0.25)"
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
                description=f"{obstacle} is hot — keep 0.25m clearance",
                level=2,
            ))
            logger.info(f"[RuleComposer] Hot-zone rule: {h_formula}")
        except Exception as e:
            logger.debug(f"[RuleComposer] Hot rule error: {e}")

    return rules
