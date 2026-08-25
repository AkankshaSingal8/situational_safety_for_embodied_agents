"""
FOL Safety Filter v6 — Multi-Prompt CoT Prompt Templates

4-phase chain:
  P1: Scene description   (free text, warm-up reasoning, all 3 views)
  P2: Role identification (JSON, N=3 majority vote)
  P3: Property attribution (JSON, N=3 majority vote per property)
  P4: Rule composition    (JSON, N=1 — synthesis step)

Designed to work with Qwen VLM (no system-prompt). Each prompt is self-contained
and includes prior context in the text body.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def _fmt_candidates(candidates: List[str]) -> str:
    return ", ".join(f'"{c}"' for c in candidates)


def _fmt_obstacles_with_props(
    obstacles: List[str],
    properties: Dict[str, List[str]],
) -> str:
    parts = []
    for obs in obstacles:
        props = properties.get(obs, [])
        prop_str = ", ".join(props) if props else "none"
        parts.append(f'  - "{obs}" (properties: {prop_str})')
    return "\n".join(parts) if parts else "  (none identified)"


# ---------------------------------------------------------------------------
# Prompt 1 — Scene Description (free text CoT warm-up)
# ---------------------------------------------------------------------------

def prompt_scene_description(
    task_description: str,
    candidates: List[str],
) -> str:
    cand_str = _fmt_candidates(candidates)
    return (
        f'You are a robot safety assistant analyzing a manipulation scene.\n\n'
        f'Task the robot is performing: "{task_description}"\n'
        f'Objects that may be present: {cand_str}\n\n'
        f'You are given images from camera views of the robot workspace.\n\n'
        f'Carefully analyze the images and describe the scene step by step:\n\n'
        f'1. Which objects can you identify? Where is each one (left/right/center, near/far)?\n'
        f'2. What is the robot gripper currently doing (reaching, grasping, holding)?\n'
        f'3. Based on the task "{task_description}", which object is the robot trying to pick up?\n'
        f'4. Which objects are NOT the task target and could be obstacles the robot should avoid?\n'
        f'5. Do any objects appear fragile (ceramic, glass)? Hot or dangerous (flame, heater)? '
        f'Containing liquid?\n\n'
        f'Be specific. Your analysis will generate robot safety rules.'
    )


# ---------------------------------------------------------------------------
# Prompt 2 — Role Identification (JSON output, N=3 majority vote)
# ---------------------------------------------------------------------------

def prompt_role_identification(
    task_description: str,
    candidates: List[str],
    scene_description: str,
) -> str:
    cand_str = _fmt_candidates(candidates)
    return (
        f'Scene analysis:\n{scene_description}\n\n'
        f'Task: "{task_description}"\n'
        f'Objects in scene: {cand_str}\n\n'
        f'Based on the scene analysis, identify each object\'s role.\n\n'
        f'Definitions:\n'
        f'- task_target: the ONE object the robot needs to pick up\n'
        f'- goal_location: where robot places the object (surface/container), or null\n'
        f'- obstacles: objects the robot must NOT touch (NOT task_target or goal_location)\n\n'
        f'Rules:\n'
        f'- task_target must be exactly ONE name from the list\n'
        f'- obstacles must NOT include task_target or goal_location\n'
        f'- Only list objects you are confident are obstacles\n\n'
        f'Respond with ONLY valid JSON, no other text:\n'
        f'{{\n'
        f'  "task_target": "exact name from list or null",\n'
        f'  "goal_location": "exact name from list or null",\n'
        f'  "obstacles": ["obstacle1", "obstacle2"]\n'
        f'}}'
    )


# ---------------------------------------------------------------------------
# Prompt 3 — Property Attribution (JSON output, N=3 majority vote)
# ---------------------------------------------------------------------------

def prompt_property_attribution(
    obstacles: List[str],
    scene_description: str,
) -> str:
    obs_str = _fmt_candidates(obstacles)
    return (
        f'Scene analysis:\n{scene_description}\n\n'
        f'Obstacles to assess: {obs_str}\n\n'
        f'For each obstacle, determine which physical safety properties apply.\n'
        f'Only assign a property if you are CONFIDENT it applies from what you observe.\n\n'
        f'Properties:\n'
        f'- IS_FRAGILE: breaks easily if hit (ceramic bowl, glass, egg, porcelain)\n'
        f'- IS_HOT: dangerous temperature (candle flame, stove burner, heater, glowing)\n'
        f'- IS_SPILLABLE: open container with liquid contents that could spill\n'
        f'- IS_SHARP: cutting edges (knife, scissors, blade, pointed tool)\n'
        f'- IS_HEAVY: very heavy (cast iron, stone, full large pot)\n\n'
        f'Respond with ONLY valid JSON, no other text:\n'
        f'{{\n'
        f'  "properties": {{\n'
        f'    "obstacle_name": ["IS_FRAGILE"],\n'
        f'    "other_obstacle": []\n'
        f'  }}\n'
        f'}}\n\n'
        f'Use empty list [] for obstacles with no applicable properties.'
    )


# ---------------------------------------------------------------------------
# Prompt 4 — Rule Composition (JSON output, N=1 synthesis)
# ---------------------------------------------------------------------------

def prompt_rule_composition(
    task_description: str,
    task_target: Optional[str],
    goal_location: Optional[str],
    obstacles: List[str],
    obstacle_properties: Dict[str, List[str]],
    scene_description: str,
) -> str:
    target_str = f'"{task_target}"' if task_target else "unknown"
    goal_str = f'"{goal_location}"' if goal_location else "none"
    obs_props_str = _fmt_obstacles_with_props(obstacles, obstacle_properties)

    return (
        f'You are writing safety constraints for a robot arm controller.\n\n'
        f'Task: "{task_description}"\n'
        f'Task target (robot picks this up): {target_str}\n'
        f'Goal location (robot places it here): {goal_str}\n'
        f'Obstacles:\n{obs_props_str}\n\n'
        f'Safety predicate syntax:\n'
        f'  NEAR(eef, OBJECT_NAME, RADIUS)  — end-effector within RADIUS meters\n'
        f'  HOLDING(eef, OBJECT_NAME)        — gripper grasping the object\n'
        f'  IS_FRAGILE(OBJECT_NAME)          — object is fragile\n'
        f'  IS_HOT(OBJECT_NAME)              — object is hot\n'
        f'  IS_SPILLABLE(OBJECT_NAME)        — object contains liquid\n\n'
        f'Safety actions:\n'
        f'  "avoid" + radius (meters): keep end-effector this far away\n'
        f'  "slow" + v_max (m/s): limit speed near object\n'
        f'  "rotation_lock": prevent wrist tilting\n\n'
        f'Parameter guidelines:\n'
        f'  IS_HOT:      avoid radius=0.25\n'
        f'  IS_FRAGILE:  slow v_max=0.08, avoid radius=0.12\n'
        f'  IS_SHARP:    avoid radius=0.12\n'
        f'  IS_HEAVY:    slow v_max=0.12\n'
        f'  default:     avoid radius=0.18\n\n'
        f'Write minimal safety rules for the obstacles above ONLY.\n'
        f'Do NOT write rules for the task target or goal location.\n'
        f'Use exact object names from the obstacles list.\n\n'
        f'Respond with ONLY valid JSON, no other text:\n'
        f'{{\n'
        f'  "rules": [\n'
        f'    {{"name": "AVOID_OBS", "when": "NEAR(eef, object_name, 0.20)", '
        f'"then": {{"action": "avoid", "radius": 0.20}}}}\n'
        f'  ],\n'
        f'  "composed_predicates": [\n'
        f'    {{"name": "CARRYING_SPILLABLE", '
        f'"definition": "IS_SPILLABLE(object_name) AND HOLDING(eef, object_name)", '
        f'"then": {{"action": "rotation_lock"}}}}\n'
        f'  ]\n'
        f'}}\n\n'
        f'Use empty list [] for "composed_predicates" if none apply.'
    )
