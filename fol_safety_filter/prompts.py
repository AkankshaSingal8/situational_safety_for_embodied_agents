"""
FOL Safety Filter v7 — Real-World Compatible Multi-Prompt VLM Pipeline

3-prompt design with N=3 majority voting:
  Prompt A: Obstacle identification  (N=3 votes, clean names only)
  Prompt B: Spatial predicates       (N=3 votes, approach directions + radius)
  Prompt C: Physical properties      (N=3 votes, commonsense object properties)

Key design principles:
  - Object names passed to VLM are clean (no _obstacle suffix, no numeric IDs)
  - VLM is told explicitly which object is the manipulated target and goal
  - No MuJoCo/sim-specific information is given to VLM
  - Pipeline is identical for sim and real robot
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Prompt A — Obstacle Identification
# Based on p5_cot_structured (best-performing without positions, 70% accuracy)
# Extended with explicit target/goal exclusion.
# ---------------------------------------------------------------------------

def prompt_obstacle_identification(
    task_description: str,
    clean_object_list: List[str],
    target_object: Optional[str] = None,
    goal_object: Optional[str] = None,
) -> str:
    obj_str = "\n".join(f"- {name}" for name in clean_object_list)
    target_line = (
        f'Object being picked up (NOT an obstacle): "{target_object}"'
        if target_object else
        "Figure out which object is being picked up from the task instruction."
    )
    goal_line = (
        f'Destination object (NOT an obstacle): "{goal_object}"'
        if goal_object else
        "Figure out the destination from the task instruction."
    )

    return (
        f'You are a robot safety assistant. Follow the reasoning steps exactly, then output JSON.\n\n'
        f'Task: "{task_description}"\n'
        f'{target_line}\n'
        f'{goal_line}\n\n'
        f'Objects visible in this scene:\n{obj_str}\n\n'
        f'STEP 1 — Confirm task objects.\n'
        f'The robot picks up the target and places it at the destination. '
        f'These two are task objects — they are NOT obstacles.\n\n'
        f'STEP 2 — Look at the images carefully.\n'
        f'Which objects from the list above can you actually see on the table?\n\n'
        f'STEP 3 — Exclude task objects.\n'
        f'Remove the manipulated object and the destination from consideration.\n\n'
        f'STEP 4 — Identify the obstacle.\n'
        f'From the remaining visible objects, choose the ONE that is closest to the robot '
        f"arm's likely path of motion and could cause a collision.\n\n"
        f'STEP 5 — Output your answer.\n'
        f'Use the exact name from the object list above.\n\n'
        f'Output ONLY valid JSON with no additional text:\n'
        f'{{\n'
        f'  "reasoning": "<write out your answers to Steps 1-5>",\n'
        f'  "object": "<exact name from the object list above>"\n'
        f'}}'
    )


# ---------------------------------------------------------------------------
# Prompt B — Spatial Predicates
# Given the identified obstacle, which directions of approach are unsafe?
# ---------------------------------------------------------------------------

def prompt_spatial_predicates(
    obstacle_name: str,
    task_description: str,
) -> str:
    return (
        f'You are a robot safety expert.\n\n'
        f'Task: "{task_description}"\n'
        f'Object the robot must avoid: "{obstacle_name}"\n\n'
        f'Look carefully at the provided images.\n\n'
        f'For the object "{obstacle_name}", determine which approach directions are UNSAFE for the robot arm.\n\n'
        f'Available direction labels (use these exact strings):\n'
        f'  "above"              — the robot should not approach from directly above\n'
        f'  "below"              — the robot should not approach from below\n'
        f'  "around in front of" — the robot should not pass in front of it\n'
        f'  "around behind"      — the robot should not pass behind it\n\n'
        f'Also estimate the safe avoidance radius in meters (how far the robot arm should stay away).\n'
        f'Typical values: 0.12m (small/precise) to 0.25m (large/dangerous).\n\n'
        f'Output ONLY valid JSON with no additional text:\n'
        f'{{\n'
        f'  "reasoning": "<why these directions are unsafe based on the object\'s position and shape>",\n'
        f'  "unsafe_directions": ["above"],\n'
        f'  "avoidance_radius_m": 0.18\n'
        f'}}'
    )


# ---------------------------------------------------------------------------
# Prompt C — Physical Properties
# Commonsense reasoning about the object's physical safety properties.
# No images needed — pure knowledge about object type.
# ---------------------------------------------------------------------------

def prompt_physical_properties(
    obstacle_name: str,
    task_description: str,
) -> str:
    return (
        f'You are a robot safety expert.\n\n'
        f'Task: "{task_description}"\n'
        f'Object: "{obstacle_name}"\n\n'
        f'Based on what you know about this type of object, answer the following questions.\n'
        f'Use your commonsense knowledge about the physical properties of "{obstacle_name}".\n\n'
        f'Questions:\n'
        f'1. IS_FRAGILE: Would it break or shatter if the robot accidentally hits it? '
        f'(e.g. glass vase, ceramic bowl, egg — yes; metal pot, book — no)\n'
        f'2. IS_SPILLABLE: Does it contain liquid that would spill if knocked over? '
        f'(e.g. moka pot with coffee, milk carton, wine bottle — yes; empty bowl, book — no)\n'
        f'3. IS_HOT: Is it at a dangerous temperature? '
        f'(e.g. lit candle, hot stove burner, freshly brewed coffee pot — yes; book, bowl — no)\n'
        f'4. IS_SHARP: Does it have sharp edges that could damage the robot or cause harm? '
        f'(e.g. knife, scissors — yes; mug, bottle — no)\n'
        f'5. REQUIRES_CAUTION: Should the robot move extra slowly near this object for any reason?\n'
        f'6. ROTATION_LOCK: Must the robot keep its gripper upright (not tilt) while near this? '
        f'(true for spillable or fragile objects that require careful approach)\n\n'
        f'Output ONLY valid JSON with no additional text:\n'
        f'{{\n'
        f'  "reasoning": "<brief explanation for each property>",\n'
        f'  "IS_FRAGILE": false,\n'
        f'  "IS_SPILLABLE": false,\n'
        f'  "IS_HOT": false,\n'
        f'  "IS_SHARP": false,\n'
        f'  "REQUIRES_CAUTION": false,\n'
        f'  "ROTATION_LOCK": false\n'
        f'}}'
    )


def prompt_rule_composition(
    task_description: str,
    obstacle_name: str,
    properties: List[str],
    target_name: Optional[str] = None,
    goal_name: Optional[str] = None,
) -> str:
    """Prompt D — compose FOL safety rules and new predicates from the
    grounded predicate vocabulary. Text-only (grounding already done by A-C)."""
    props_str = ", ".join(properties) if properties else "none detected"
    target_str = f'"{target_name}"' if target_name else "unknown"
    goal_str = f'"{goal_name}"' if goal_name else "none"

    return (
        f'You are writing safety constraints for a robot arm controller.\n\n'
        f'Task: "{task_description}"\n'
        f'Task target (robot picks this up): {target_str}\n'
        f'Goal location (robot places it here): {goal_str}\n'
        f'Obstacle to avoid: "{obstacle_name}" (properties: {props_str})\n\n'
        f'Grounded predicate vocabulary:\n'
        f'  NEAR(eef, OBJECT_NAME, RADIUS)  — end-effector within RADIUS meters\n'
        f'  HOLDING(eef, OBJECT_NAME)        — gripper grasping the object\n'
        f'  IS_FRAGILE(OBJECT_NAME)          — object is fragile\n'
        f'  IS_HOT(OBJECT_NAME)              — object is hot\n'
        f'  IS_SPILLABLE(OBJECT_NAME)        — object contains liquid\n\n'
        f'Safety actions:\n'
        f'  "avoid" + radius (meters): keep end-effector this far away\n'
        f'  "slow" + v_max (m/s): limit speed when the rule fires\n'
        f'  "rotation_lock": prevent wrist tilting\n\n'
        f'Parameter guidelines:\n'
        f'  IS_HOT:      avoid radius=0.25\n'
        f'  IS_FRAGILE:  slow v_max=0.08\n'
        f'  IS_SHARP:    avoid radius=0.15\n'
        f'  default:     avoid radius=0.20\n\n'
        f'Write minimal safety rules for the obstacle above ONLY.\n'
        f'Do NOT write rules for the task target or goal location.\n'
        f'Use the exact obstacle name "{obstacle_name}" in formulas.\n'
        f'You may COMPOSE new predicates by combining vocabulary predicates '
        f'with AND when the situation warrants (e.g. slow down only while '
        f'holding something near a fragile obstacle).\n\n'
        f'Respond with ONLY valid JSON, no other text:\n'
        f'{{\n'
        f'  "rules": [\n'
        f'    {{"name": "AVOID_OBSTACLE", "when": "NEAR(eef, {obstacle_name}, 0.20)", '
        f'"then": {{"action": "avoid", "radius": 0.20}}}}\n'
        f'  ],\n'
        f'  "composed_predicates": [\n'
        f'    {{"name": "CARRYING_NEAR_OBSTACLE", '
        f'"definition": "HOLDING(eef, {target_name or "object"}) AND NEAR(eef, {obstacle_name}, 0.25)", '
        f'"then": {{"action": "slow", "v_max": 0.08}}}}\n'
        f'  ]\n'
        f'}}\n\n'
        f'Use empty list [] for "composed_predicates" if none apply.'
    )
