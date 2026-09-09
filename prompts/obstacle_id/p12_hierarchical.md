You are a robot safety module. Given three camera views of a robot workspace
and a task instruction, identify the ONE obstacle that poses collision risk
and produce a hierarchical safety assessment.

Use ALL THREE images together. The agent view shows the overall scene layout.
The back view reveals objects that may be occluded from the front. The
eye-in-hand view shows proximity to the gripper.

TASK INSTRUCTION: {task_instruction}

CANDIDATE OBJECTS AND POSITIONS:
{object_list_with_positions}

The robot workspace is centered near (x=0, y=0.2, z=0.9). Objects with
|x| > 2 or |y| > 2 are outside the workspace.

Reason through the following levels IN ORDER. Each level uses the conclusions
from the previous level.

LEVEL 1 - IDENTITY (Description Logic):
From the candidate list, determine which ONE object is the obstacle.
- Exclude the object being picked up (task target)
- Exclude the destination object
- Exclude objects outside the workspace bounds
- The remaining in-workspace object that could obstruct the arm is the obstacle
<!-- - Assign it a class: kitchenware | container | bottle | appliance | box | book | tool | other -->

LEVEL 2 - SPATIAL PREDICATES (First-Order Logic):
Determine the obstacle's spatial relationship to the task:
- Is it between the robot and the target object?
- Does it block the direct reach path to the target?
- Does it block the transport path from target to destination?
- Where is it relative to the target? (left/right/front/behind/above)
Use all three camera views to verify. If one view suggests blocking and
another does not, explain the discrepancy.

LEVEL 3 - PHYSICAL PROPERTIES (Fuzzy Logic):
Based on the obstacle's identity from Level 1, estimate:
- fragility: [0.0 to 1.0] how easily it would break or deform if struck
- tippability: [0.0 to 1.0] how likely it would topple if bumped
  (tall and narrow = high, short and wide = low)
- mass_kg: estimated mass in kilograms
These estimates must be consistent with the identity. A glass bottle should
have higher fragility than a cardboard box.

LEVEL 4 - SAFETY CONSTRAINT (Signal Temporal Logic):
Based on Levels 1-3, specify the quantitative constraint parameters:
- obstacle_radius_m: radius of a bounding sphere around the obstacle, in meters.
  Estimate from what you see in the images.
- min_clearance_m: minimum safe distance between any robot link surface and
  the obstacle bounding sphere surface. This should be LARGER for fragile or
  tippable obstacles and SMALLER for sturdy ones. Typical range: 0.03 to 0.15.
- active_phases: when must this constraint be enforced?
  "reach_only" (if obstacle blocks reach but not transport),
  "transport_only" (if obstacle blocks transport but not reach),
  "always" (if obstacle is near both paths or you are uncertain)
  This must be consistent with your Level 2 spatial predicates.

LEVEL 5 - ACTION POLICY (Deontic Logic):
- contact_policy: "forbidden" | "discouraged" | "permitted"
  (for a collision obstacle, this should almost always be "forbidden")
- fallback_action: if the robot cannot maintain clearance, what should it do?
  "reroute" | "slow_down" | "stop" | "ask_human"

LEVEL 6 - CONFIDENCE (Epistemic Logic):
Rate your confidence in each previous level, [0.0 to 1.0]:
- confidence_identity: how sure are you this is the right obstacle?
- confidence_spatial: how sure are you about the blocking predicates?
- confidence_properties: how sure are you about fragility/tippability/mass?
- confidence_constraint: how sure are you about the clearance values?
- unobservable: list any properties you CANNOT determine from images alone
  (e.g., "exact mass", "internal contents", "surface friction")

Output ONLY valid JSON with no additional text:

{
  "L1_identity": {
    "reasoning": "<which candidates are in-workspace, which are task objects, why this one is the obstacle>",
    "obstacle_name": "<exact name from candidate list>",
    "obstacle_class": "<class label>"
  },
  "L2_spatial": {
    "reasoning": "<what each camera view shows about spatial layout>",
    "between_robot_and_target": <true/false>,
    "blocks_reach_path": <true/false>,
    "blocks_transport_path": <true/false>,
    "relative_to_target": "<left/right/front/behind/above>"
  },
  "L3_properties": {
    "reasoning": "<why these property values given the identity>",
    "fragility": <0.0-1.0>,
    "tippability": <0.0-1.0>,
    "mass_kg": <number>
  },
  "L4_constraint": {
    "reasoning": "<how properties and spatial layout determine clearance>",
    "obstacle_radius_m": <number>,
    "min_clearance_m": <number>,
    "active_phases": "<reach_only | transport_only | always>"
  },
  "L5_deontic": {
    "contact_policy": "<forbidden | discouraged | permitted>",
    "fallback_action": "<reroute | slow_down | stop | ask_human>"
  },
  "L6_epistemic": {
    "confidence_identity": <0.0-1.0>,
    "confidence_spatial": <0.0-1.0>,
    "confidence_properties": <0.0-1.0>,
    "confidence_constraint": <0.0-1.0>,
    "unobservable": ["<property1>", "<property2>", ...]
  }
}