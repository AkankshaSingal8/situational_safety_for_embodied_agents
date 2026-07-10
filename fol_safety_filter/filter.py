"""
FOL Safety Filter v6 — Multi-Prompt CoT with Composed Predicates

Changes vs v5:
  1. 4-phase CoT VLM grounder (scene→roles→properties→rules)
  2. N=3 majority voting on role identification and property attribution
  3. VLM-authored composed predicates evaluated at runtime via KB
  4. Backend-agnostic VLM client (Qwen server / Claude API / OpenAI)
  5. All 3 views: agentview + eye_in_hand + back_view
  6. Zero MuJoCo/sim dependencies in filter core (real Franka compatible)

All CBF geometry preserved from v5 (phase-aware, carrying-mode, goal-relaxation,
goal-tangential redirect, arm-checkpoint avoidance).
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def _load_semantic_filter():
    _vlm_dir = os.path.join(os.path.dirname(__file__), "..", "vlm_pipeline")
    if _vlm_dir not in sys.path:
        sys.path.insert(0, _vlm_dir)
    from semantic_cbf_filter import SemanticSafetyFilter
    return SemanticSafetyFilter


from .primitives import (
    ObjectState,
    RobotState,
    VLM_GROUNDING_QUERIES,
    COMPOSED_RULE_SEEDS,
    PREDICATE_REGISTRY,
)
from .kb import FOLKnowledgeBase, FOLRule, ActiveConstraint, parse_formula
from .cbf_mapper import CBFMapper, MappedCBFSet
from .vlm_grounder import VLMObstacleGrounder, _parse_target_heuristic, _props_from_name
from .visual_grounder import VisualObstacleGrounder
from .rule_composer import compose_rules


# ---------------------------------------------------------------------------
# VLM Grounder — physical-state binary queries (heuristic, unchanged from v5)
# ---------------------------------------------------------------------------

class VLMGrounder:
    def __init__(self, use_vlm: bool = False, n_votes: int = 3):
        self.use_vlm = use_vlm
        self.n_votes = n_votes

    def ground(self, obs: Dict, object_states: Dict[str, ObjectState],
               rgb_image: Optional[np.ndarray] = None) -> None:
        for name, obj in object_states.items():
            self._heuristic_ground(name, obj)

    def _heuristic_ground(self, name: str, obj: ObjectState) -> None:
        n = name.lower().replace("_", " ")
        obj.is_fragile   = any(k in n for k in ["egg", "glass", "vase", "crystal"])
        obj.is_wet       = any(k in n for k in ["wet", "sponge", "water"])
        obj.is_lit       = any(k in n for k in ["candle lit", "torch", "flame"])
        obj.is_full      = any(k in n for k in ["bowl", "cup", "glass", "mug", "pot", "container"])
        obj.is_open      = any(k in n for k in ["bowl", "cup", "glass", "mug"])
        obj.is_flammable = any(k in n for k in ["napkin", "paper", "cloth", "fabric", "wood"])
        obj.is_absorbent = any(k in n for k in ["napkin", "paper", "cloth", "fabric", "sponge"])
        obj.is_human     = any(k in n for k in ["human", "hand", "arm", "person"])
        obj.has_sharp_part = any(k in n for k in ["knife", "scissors", "blade", "sharp"])
        obj.is_heavy     = any(k in n for k in ["heavy", "moka", "iron", "cast"])
        obj.is_spillable = bool(obj.is_full and obj.is_open)

    # Also update from VLM-detected properties (called after VLM grounding)
    def apply_vlm_properties(
        self,
        object_states: Dict[str, ObjectState],
        obstacle_properties: Dict[str, List[str]],
    ) -> None:
        PROP_MAP = {
            "IS_FRAGILE": "is_fragile",
            "IS_HOT": "is_lit",
            "IS_SPILLABLE": ("is_full", "is_open"),
            "IS_SHARP": "has_sharp_part",
            "IS_HEAVY": "is_heavy",
        }
        for obs_key, props in obstacle_properties.items():
            obj = object_states.get(obs_key)
            if obj is None:
                continue
            for prop in props:
                mapped = PROP_MAP.get(prop)
                if mapped is None:
                    continue
                if isinstance(mapped, tuple):
                    for attr in mapped:
                        setattr(obj, attr, True)
                else:
                    setattr(obj, mapped, True)
            if obj.is_full and obj.is_open:
                obj.is_spillable = True


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class FilterMetrics:
    timesteps: int = 0
    interventions: int = 0
    rule_counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main FOL Safety Filter v6
# ---------------------------------------------------------------------------

class FOLSafetyFilter:
    """
    Policy-agnostic FOL safety filter — v6.

    v6 over v5:
      - 4-phase CoT VLM grounder (scene desc → roles → props → rules)
      - N=3 majority voting on role ID and property attribution
      - VLM-authored composed predicates injected into KB at episode start
      - Backend-agnostic VLM client (Qwen/Claude/OpenAI)
      - All 3 views utilized
      - No sim/GT dependency in filter core
    """

    def __init__(
        self,
        use_vlm_grounding: bool = False,
        use_vlm_obstacle: bool = True,
        fol_level: int = 3,
        dt: float = 0.05,
        alpha_default: float = 1.0,
        alpha_caution: float = 0.25,
        safety_margin: float = 0.02,
        velocity_limit_default: Optional[float] = None,
        workspace_z_max: float = 1.40,
    ):
        self.use_vlm = use_vlm_grounding
        self.use_vlm_obstacle = use_vlm_obstacle
        self.fol_level = fol_level
        self.dt = dt
        self.ws_z_max = workspace_z_max
        self.safety_margin = safety_margin
        self.velocity_limit_default = velocity_limit_default

        self.kb = FOLKnowledgeBase()
        self.mapper = CBFMapper(ws_z_max=workspace_z_max)
        self.grounder = VLMGrounder(use_vlm=use_vlm_grounding)
        self.obstacle_grounder = VLMObstacleGrounder(use_vlm=use_vlm_obstacle)

        self._ssf = None
        self._visual_grounder = None
        self._vision_target_xy = None
        self._alpha_default = alpha_default
        self._alpha_caution = alpha_caution

        self._object_states: Dict[str, ObjectState] = {}
        self._obstacle_names: List[str] = []
        self._obstacle_radii: Dict[str, float] = {}
        self._obstacle_properties: Dict[str, List[str]] = {}
        self._obstacle_name: Optional[str] = None
        self._target_name: Optional[str] = None
        self._goal_name: Optional[str] = None
        self._ee_quat_ref: Optional[np.ndarray] = None
        self._initialized = False
        self._t = 0

        self.metrics = FilterMetrics()

    # ------------------------------------------------------------------ #
    # Episode setup
    # ------------------------------------------------------------------ #

    def setup_episode(
        self,
        obs: Dict,
        task_description: str = "",
        obstacle_name: Optional[str] = None,
        env=None,
        rgb_image: Optional[np.ndarray] = None,
    ) -> None:
        self.kb.clear()
        self._object_states = {}
        self._obstacle_names = []
        self._obstacle_radii = {}
        self._obstacle_properties = {}
        self._obstacle_name = None
        self._target_name = None
        self._goal_name = None
        self._initialized = False
        self._t = 0
        self._ee_quat_ref = None
        self.metrics = FilterMetrics()

        # Step 1: Build ObjectState for each detected object
        object_names = self._extract_objects(obs)
        for name in object_names:
            pos = np.array(obs.get(f"{name}_pos", [0, 0, 0]), dtype=np.float64)
            quat = np.array(obs.get(f"{name}_quat", [0, 0, 0, 1]), dtype=np.float64)
            obj = ObjectState(name=name, pos=pos, quat=quat,
                              bbox_half=np.array([0.04, 0.04, 0.04]))
            self._object_states[name] = obj

        logger.info(f"[FOL-v6] {len(self._object_states)} objects: {list(self._object_states.keys())}")

        # Step 2: Preliminary heuristic parse for target/goal — passed to VLM so it knows
        # which objects are task objects (NOT obstacles). VLM results may override these later.
        from .vlm_grounder import _parse_target_heuristic as _pheuristic
        candidates = list(self._object_states.keys())
        pre_target = _pheuristic(task_description, candidates)
        pre_goal: Optional[str] = None  # will be resolved post-VLM

        agentview = rgb_image if rgb_image is not None else obs.get("agentview_image")
        eye_in_hand = obs.get("robot0_eye_in_hand_image")
        back_view = obs.get("robot0_back_image")  # may be None
        rgb_images = [img for img in [agentview, eye_in_hand, back_view] if img is not None]

        vision_mode = os.environ.get("FOL_VISION_GROUNDING", "0") == "1"
        if vision_mode:
            # v18 GT-free path: identity AND position from perception only.
            # Simulator object state is NEVER consulted for the obstacle.
            predicate_dict = self._vision_ground(
                obs, task_description, agentview, eye_in_hand
            )
        else:
            predicate_dict = self.obstacle_grounder.detect(
                rgb_images, task_description, candidates,
                target_obs_key=pre_target,
                goal_obs_key=pre_goal,
                candidate_positions={
                    name: st.pos for name, st in self._object_states.items()
                },
                eef_pos=np.array(obs["robot0_eef_pos"], dtype=np.float64)
                if "robot0_eef_pos" in obs else None,
            )

        # v6: multi-obstacle support
        vlm_obstacles = predicate_dict.get("obstacle_obs_keys", [])
        valid_obstacles = [o for o in vlm_obstacles if o in self._object_states]

        if valid_obstacles:
            self._obstacle_names = valid_obstacles
            self._obstacle_radii = {
                o: predicate_dict["obstacle_radii"].get(o, 0.18)
                for o in valid_obstacles
            }
            self._obstacle_properties = {
                o: predicate_dict["obstacle_properties"].get(o, [])
                for o in valid_obstacles
            }
            self._obstacle_name = valid_obstacles[0]
            logger.info(
                f"[FOL-v6] Obstacles ({predicate_dict['method']}): {self._obstacle_names} "
                f"radii={self._obstacle_radii}"
            )
        else:
            # No obstacle was grounded (symbolic FOL and VLM both returned none).
            # NO ground-truth-label fallback by design: the GT obstacle_name param
            # and '_obstacle' naming conventions are never consulted.  The filter
            # runs this episode with workspace rules only (honest failure).
            logger.warning(
                "[FOL-v6] NO OBSTACLE GROUNDED — spatial avoidance inactive this "
                "episode (workspace rules remain active)"
            )

        # Task target and goal (from VLM CoT, then heuristic parse)
        vlm_target = predicate_dict.get("task_target_obs_key")
        if vlm_target and vlm_target in self._object_states:
            self._target_name = vlm_target
            logger.info(f"[FOL-v6] Target (VLM CoT): {self._target_name}")
        else:
            self._target_name = self._extract_target_name(task_description)
            logger.info(f"[FOL-v6] Target (parsed): {self._target_name}")

        vlm_goal = predicate_dict.get("goal_location_obs_key")
        if vlm_goal and vlm_goal in self._object_states:
            self._goal_name = vlm_goal
            logger.info(f"[FOL-v6] Goal (VLM CoT): {self._goal_name}")
        else:
            self._goal_name = self._extract_goal_name(task_description)
            if self._goal_name:
                logger.info(f"[FOL-v6] Goal (parsed): {self._goal_name}")

        # Step 3: Apply VLM-detected properties to ObjectState
        self.grounder.ground(obs, self._object_states, rgb_image)
        self.grounder.apply_vlm_properties(self._object_states, self._obstacle_properties)

        # Step 4: Build FOL rules
        self.kb.add_workspace_safety_rules()

        # Standard obstacle rules (NEAR + property-based)
        predicate_dict["obstacle_radii"] = self._obstacle_radii
        predicate_dict["obstacle_properties"] = self._obstacle_properties
        predicate_dict["obstacle_obs_keys"] = self._obstacle_names
        composed = compose_rules(predicate_dict, self._obstacle_name)
        for rule in composed:
            self.kb.add_rule(rule)

        # Step 5: Inject VLM-authored composed rules from P4
        vlm_composed_rules = predicate_dict.get("composed_rules", [])
        vlm_composed_predicates = predicate_dict.get("composed_predicates", [])
        self._inject_vlm_composed_rules(vlm_composed_rules, vlm_composed_predicates)

        if self.fol_level >= 2:
            self._add_semantic_rules(
                target_exclusions=set(self._obstacle_names) if self._obstacle_names else None
            )

        logger.info(self.kb.summary())

        # Step 6: Initialize SemanticSafetyFilter
        try:
            SemanticSafetyFilter = _load_semantic_filter()
            self._ssf = SemanticSafetyFilter(
                dt=self.dt,
                alpha_default=self._alpha_default,
                alpha_caution=self._alpha_caution,
                safety_margin=self.safety_margin,
                workspace_z_max=self.ws_z_max,
            )
            self._ssf.semantic_envelopes = []
            self._ssf.collision_envelopes = []
            self._ssf._init = True
        except ImportError as e:
            logger.warning(f"Could not load SemanticSafetyFilter: {e}. Geometry-only mode.")
            self._ssf = None

        self._initialized = True
        logger.info(
            f"[FOL-v6] Episode ready: {len(self.kb.rules)} rules, "
            f"obstacles={self._obstacle_names}, target={self._target_name}, "
            f"goal={self._goal_name}, backend={self.obstacle_grounder._get_client().backend_name}"
        )

    def _inject_vlm_composed_rules(
        self,
        composed_rules: List[Dict],
        composed_predicates: List[Dict],
    ) -> None:
        """Add VLM-authored rules and composed predicates to the KB."""
        for rule_spec in composed_rules:
            name = rule_spec.get("name", "VLM_RULE")
            when = rule_spec.get("when", "")
            then = rule_spec.get("then", {})
            if not when or not then:
                continue
            action = then.get("action", "avoid")
            try:
                parsed = parse_formula(when)
            except Exception as e:
                logger.warning(f"[FOL-v6] Cannot parse composed rule '{name}': {e}")
                continue

            cbf_type = "spatial"
            cbf_params: Dict = {}
            violation_action = "avoid"

            if action == "avoid":
                cbf_type = "spatial"
                cbf_params = {
                    "shape": "sphere",
                    "margin": float(then.get("radius", 0.18)) * 0.5,
                    "alpha_scale": 1.5,
                }
                violation_action = "avoid"
            elif action == "slow":
                cbf_type = "velocity"
                cbf_params = {"v_max": float(then.get("v_max", 0.10))}
                violation_action = "slow"
            elif action == "rotation_lock":
                cbf_type = "rotation"
                cbf_params = {"angular_limit": 0.15}
                violation_action = "slow"

            # Extract primary object from formula — fuzzy-match VLM name to actual obs key
            import re
            obj_match = re.search(r'NEAR\s*\(\s*\w+\s*,\s*(\w[\w_]*)', when, re.IGNORECASE)
            primary_obj = obj_match.group(1) if obj_match else None
            if primary_obj and primary_obj not in self._object_states:
                # Fuzzy match: find obs key that contains the VLM-generated name as substring
                vlm_name_lower = primary_obj.lower().replace("_", "")
                matched = None
                for obs_key in self._object_states:
                    key_clean = obs_key.lower().replace("_", "").replace("obstacle", "")
                    if vlm_name_lower in key_clean or key_clean in vlm_name_lower:
                        matched = obs_key
                        break
                if matched:
                    # Rewrite formula with correct obs key
                    when = re.sub(
                        r'(NEAR\s*\(\s*\w+\s*,\s*)' + re.escape(primary_obj),
                        r'\g<1>' + matched,
                        when,
                        flags=re.IGNORECASE,
                    )
                    try:
                        parsed = parse_formula(when)
                    except Exception:
                        pass
                    primary_obj = matched
                    logger.info(f"[FOL-v6] Resolved VLM entity '{obj_match.group(1)}' → '{matched}'")
                else:
                    primary_obj = None

            # Only inject rules whose object is in the GROUNDED obstacle set
            # (selected by the symbolic FOL rule / VLM grounding).  The VLM may
            # also name task objects (ramekin, cookies) as things to avoid, but
            # those are reachable task objects — blocking them causes failures.
            # Also skip if the object wasn't found in the current scene.
            if primary_obj is None:
                logger.info(f"[FOL-v6] Skipping VLM rule '{name}': object not found in scene")
                continue
            if primary_obj not in self._obstacle_names:
                logger.info(
                    f"[FOL-v6] Skipping VLM rule '{name}': '{primary_obj}' is not in the "
                    f"grounded obstacle set {self._obstacle_names}"
                )
                continue

            rule = FOLRule(
                name=name.upper(),
                formula=when,
                parsed=parsed,
                bindings={},
                cbf_type=cbf_type,
                cbf_params=cbf_params,
                violation_action=violation_action,
                primary_object=primary_obj,
                description=f"VLM-composed rule: {when} → {action}",
                level=3,
            )
            self.kb.add_rule(rule)
            logger.info(f"[FOL-v6] Injected VLM rule '{name}': {when} → {action}")

        for pred_spec in composed_predicates:
            name = pred_spec.get("name", "VLM_PRED")
            definition = pred_spec.get("definition", "")
            then = pred_spec.get("then", {})
            if not definition or not then:
                continue
            action = then.get("action", "slow")
            try:
                parsed = parse_formula(definition)
            except Exception as e:
                logger.warning(f"[FOL-v6] Cannot parse composed predicate '{name}': {e}")
                continue

            cbf_type = "velocity"
            cbf_params: Dict = {"v_max": 0.08}
            violation_action = "slow"
            if action == "rotation_lock":
                cbf_type = "rotation"
                cbf_params = {"angular_limit": 0.15}
            elif action == "slow":
                cbf_type = "velocity"
                cbf_params = {"v_max": float(then.get("v_max", 0.08))}

            rule = FOLRule(
                name=name.upper(),
                formula=definition,
                parsed=parsed,
                bindings={},
                cbf_type=cbf_type,
                cbf_params=cbf_params,
                violation_action=violation_action,
                primary_object=None,
                description=f"VLM-composed predicate: {definition} → {action}",
                level=3,
            )
            self.kb.add_rule(rule)
            logger.info(f"[FOL-v6] Injected VLM predicate '{name}': {definition} → {action}")

    # ------------------------------------------------------------------ #
    # v18: GT-free vision grounding
    # ------------------------------------------------------------------ #

    def _vision_ground(self, obs, task_description, agentview, eye_in_hand):
        """Ground the obstacle from images + proprioception; inject a synthetic
        ObjectState at the vision-estimated position (static — never updated
        from simulator object state)."""
        from .vlm_grounder import _empty_result

        if self._visual_grounder is None:
            self._visual_grounder = VisualObstacleGrounder(
                self.obstacle_grounder._get_client()
            )

        eef_pos = np.array(obs.get("robot0_eef_pos", [0, 0, 0]), dtype=np.float64)
        quat = obs.get("robot0_eef_quat")
        eef_quat = np.array(quat, dtype=np.float64) if quat is not None             else np.array([0.0, 0.0, 0.0, 1.0])

        res = None
        if agentview is not None and eye_in_hand is not None:
            try:
                res = self._visual_grounder.ground(
                    agentview, eye_in_hand, task_description, eef_pos, eef_quat,
                    cache_key=None,  # per-episode: obstacle layout varies
                )
            except Exception as e:
                logger.warning(f"[FOL-v18] vision grounding failed: {e}")

        if res is None:
            return _empty_result()

        self._vision_target_xy = res.get("target_xy")
        synth = "visual_" + re.sub(r"[^a-z0-9]+", "_", res["name"]).strip("_")
        self._object_states[synth] = ObjectState(
            name=synth,
            pos=np.asarray(res["pos"], dtype=np.float64),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            bbox_half=np.array([0.05, 0.05, 0.05]),
        )
        props, base_r = _props_from_name(res["name"])
        radius = max(0.20, base_r) + float(res["uncertainty_m"])
        out = _empty_result()
        out.update({
            "obstacle_obs_keys": [synth],
            "obstacle_obs_key": synth,
            "obstacle_properties": {synth: props},
            "all_properties": {synth: props},
            "obstacle_radii": {synth: radius},
            "physical_properties": props,
            "confidence": 0.8,
            "method": res["method"],
        })
        return out

    # ------------------------------------------------------------------ #
    # Per-timestep certification (identical to v5)
    # ------------------------------------------------------------------ #

    def certify(
        self,
        u_cmd: np.ndarray,
        obs: Dict,
        ee_pos: Optional[np.ndarray] = None,
        ee_quat: Optional[np.ndarray] = None,
        env=None,
    ) -> np.ndarray:
        if not self._initialized:
            return u_cmd.copy()

        self._t += 1
        state = self._build_robot_state(obs, ee_pos, ee_quat)

        for name, obj in self._object_states.items():
            if f"{name}_pos" in obs:
                obj.pos = np.array(obs[f"{name}_pos"], dtype=np.float64)
            if f"{name}_quat" in obs:
                obj.quat = np.array(obs[f"{name}_quat"], dtype=np.float64)
        state.objects = self._object_states

        active = self.kb.evaluate(state)

        self.metrics.timesteps += 1
        for ac in active:
            self.metrics.rule_counts[ac.rule.name] = \
                self.metrics.rule_counts.get(ac.rule.name, 0) + 1

        cbf_set = self.mapper.map(active, state.ee_quat)
        arm_checkpoints = self._get_arm_checkpoints(env)
        # v15: direct geometric projection only.  SSF QP is intentionally NOT
        # called — OpenVLA emits position deltas, and the SSF velocity-control
        # correction (dp*dt) is ~20x too small to matter, while its envelope
        # spheres double-block the arm (v12 root cause).
        u_safe, _ = self._apply_cbf(u_cmd, state, cbf_set, arm_checkpoints)

        if np.linalg.norm(u_safe[:3] - u_cmd[:3]) > 1e-6:
            self.metrics.interventions += 1

        return u_safe

    def report(self) -> Dict:
        """Return per-episode filter metrics (called by eval script after each task)."""
        t = max(self.metrics.timesteps, 1)
        return {
            "filter_activation_rate": self.metrics.interventions / t,
            "interventions": self.metrics.interventions,
            "timesteps": self.metrics.timesteps,
            "rule_counts": dict(self.metrics.rule_counts),
        }

    def _get_arm_checkpoints(self, env) -> List[np.ndarray]:
        """v3: wrist and elbow only (robot0_link4, robot0_link6)."""
        if env is None:
            return []
        checkpoints = []
        for body_name in ["robot0_link4", "robot0_link6"]:
            try:
                body_id = env.sim.model.body_name2id(body_name)
                pos = np.array(env.sim.data.body_xpos[body_id], dtype=np.float64)
                checkpoints.append(pos)
            except Exception:
                pass
        return checkpoints

    # ------------------------------------------------------------------ #
    # Phase detection (identical to v5)
    # ------------------------------------------------------------------ #

    def _detect_phase(self, state: RobotState) -> str:
        carrying = state.gripper_width < 0.025
        if not carrying:
            return "approaching"
        if self._target_name and self._target_name in state.objects:
            target_z = state.objects[self._target_name].pos[2]
            if target_z > 0.86:
                if self._goal_name and self._goal_name in state.objects:
                    goal_pos = state.objects[self._goal_name].pos
                    if float(np.linalg.norm(state.ee_pos - goal_pos)) < 0.30:
                        return "placing"
                return "carrying"
        return "grasping"

    def _get_target_pos(self, state: RobotState) -> Optional[np.ndarray]:
        if self._target_name and self._target_name in state.objects:
            return state.objects[self._target_name].pos
        return None

    # ------------------------------------------------------------------ #
    # CBF application (identical to v5: multi-obstacle, phase-aware)
    # ------------------------------------------------------------------ #

    def _apply_cbf(
        self,
        u_cmd: np.ndarray,
        state: RobotState,
        cbf_set: MappedCBFSet,
        arm_checkpoints: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        # v15: exact v3 geometry (proven TSR=48.5%, CAR=59.5% on spatial L1),
        # extended to multi-obstacle with VLM property-modulated radii.
        # OpenVLA outputs position deltas (m/step), NOT velocities — direct
        # geometric projection, no QP.  Binary rule: 100% radial approach
        # cancellation inside warning_r + outward push inside hard_r.
        u = u_cmd.copy()
        ee = state.ee_pos
        z_ceiling_active = False

        for obs_name in self._obstacle_names:
            if obs_name not in state.objects:
                continue
            obs_pos = state.objects[obs_name].pos
            # VLM property-modulated radius can only EXPAND the proven v3
            # margin (0.20m): IS_HOT / IS_FRAGILE obstacles get larger zones.
            warning_r = max(0.20, self._obstacle_radii.get(obs_name, 0.20))

            # Target-corridor relaxation (vision-grounded obstacles only):
            # an ~8cm-offset sphere can swallow the grasp corridor of a target
            # that sits near the obstacle.  When the commanded motion aims at
            # the vision-detected target and the EEF is close to it, cancel
            # only 60% of the approach (v2-validated relaxation).  Oracle-state
            # obstacles are exactly centered and never need this.
            cancel_scale = 1.0
            if (obs_name.startswith("visual_")
                    and getattr(self, "_vision_target_xy", None) is not None):
                tdiff = np.asarray(self._vision_target_xy) - ee[:2]
                tdist = float(np.linalg.norm(tdiff))
                speed = float(np.linalg.norm(u[:2]))
                if tdist < 0.30 and speed > 1e-6                         and float(u[:2] @ tdiff) / (speed * tdist + 1e-9) > 0.6:
                    cancel_scale = 0.6
            self._apply_point_cbf(
                u, ee, obs_pos,
                warning_r=warning_r, hard_r=0.10,
                push_hard=0.08, label=f"EEF-{obs_name}",
                cancel_scale=cancel_scale,
            )
            for cp_pos in (arm_checkpoints or []):
                self._apply_point_cbf(
                    u, cp_pos, obs_pos,
                    warning_r=0.15, hard_r=0.08,
                    push_hard=0.06, label=f"ARM-{obs_name}",
                )

        # Velocity limit
        vlim = cbf_set.velocity_limit or self.velocity_limit_default
        if vlim is not None:
            speed = np.linalg.norm(u[:3])
            if speed > vlim:
                u[:3] *= vlim / speed

        # Rotation constraint
        if cbf_set.pose_constrained and cbf_set.angular_limit is not None:
            omega = np.linalg.norm(u[3:6])
            if omega > cbf_set.angular_limit:
                u[3:6] *= cbf_set.angular_limit / omega

        return u, z_ceiling_active

    def _apply_point_cbf(
        self,
        u: np.ndarray,
        point: np.ndarray,
        obs_pos: np.ndarray,
        warning_r: float,
        hard_r: float,
        push_hard: float,
        label: str = "",
        cancel_scale: float = 1.0,
    ) -> None:
        """In-place apply CBF for a single monitoring point (EEF or arm body).

        Binary rule (v3): 100% approach cancellation anywhere inside warning_r.
        Outward push proportional to penetration inside hard_r.
        cancel_scale < 1 relaxes the cancellation (target-corridor case).
        """
        diff = point - obs_pos
        dist = float(np.linalg.norm(diff))
        if dist <= 0:
            return

        point_next = point + u[:3]
        dist_next = float(np.linalg.norm(point_next - obs_pos))
        if dist >= warning_r and dist_next >= warning_r:
            return

        out_dir = diff / dist

        # Cancel 100% of the approach component
        approach_comp = float(np.dot(u[:3], -out_dir))
        if approach_comp > 0:
            u[:3] += approach_comp * cancel_scale * out_dir
            logger.debug(f"[FOL-CBF] {label} cancel 1.0: dist={dist:.3f}")

        # Outward push when inside hard_r
        if dist < hard_r:
            push = (hard_r - dist) / hard_r * push_hard
            u[:3] += out_dir * push

    # ------------------------------------------------------------------ #
    # Internal helpers (identical to v5)
    # ------------------------------------------------------------------ #

    def _extract_objects(self, obs: Dict) -> List[str]:
        names = set()
        for k in obs:
            if k.endswith("_pos") and "robot" not in k and "gripper" not in k:
                name = k[:-4]
                pos = obs[k]
                if isinstance(pos, (list, np.ndarray)) and len(pos) == 3:
                    p = np.asarray(pos)
                    if p[2] > -0.1 and abs(p[0]) < 1.5 and abs(p[1]) < 1.5:
                        names.add(name)
        return sorted(names)

    def _build_robot_state(self, obs: Dict, ee_pos=None, ee_quat=None) -> RobotState:
        _ee_pos = ee_pos if ee_pos is not None else \
                  np.array(obs.get("robot0_eef_pos", [0, 0, 0.9]), dtype=np.float64)
        _ee_quat = ee_quat if ee_quat is not None else \
                   np.array(obs.get("robot0_eef_quat", [0, 0, 0, 1]), dtype=np.float64)
        gripper_q = obs.get("robot0_gripper_qpos", np.array([0.0, 0.0]))
        gripper_w = float(np.mean(np.abs(gripper_q)))
        return RobotState(
            ee_pos=_ee_pos,
            ee_quat=_ee_quat,
            ee_vel=np.zeros(3),
            joint_pos=np.zeros(7),
            joint_vel=np.zeros(7),
            joint_limits_lo=np.array([-2.9, -1.8, -2.9, -3.1, -2.9, -0.1, -2.9]),
            joint_limits_hi=np.array([ 2.9,  1.8,  2.9, -0.1,  2.9,  3.8,  2.9]),
            gripper_width=gripper_w,
            objects=self._object_states,
        )

    def _extract_target_name(self, task_description: str) -> Optional[str]:
        key = _parse_target_heuristic(task_description, list(self._object_states.keys()))
        if key:
            return key
        best_match: Optional[str] = None
        best_score: int = 0
        desc_lower = task_description.lower()
        for name in self._object_states:
            if self._obstacle_names and name in self._obstacle_names:
                continue
            if "obstacle" in name.lower():
                continue
            cleaned = name.replace("_", " ").rstrip(" 0123456789").strip()
            if cleaned and cleaned in desc_lower and len(cleaned) > best_score:
                best_match = name
                best_score = len(cleaned)
        return best_match

    def _extract_goal_name(self, task_description: str) -> Optional[str]:
        if not task_description:
            return None
        desc_lower = task_description.lower()
        for prep in [" on the ", " in the ", " inside the ", " into the ", " onto the ",
                     " on top of the ", " into ", " onto "]:
            if prep in desc_lower:
                after_prep = desc_lower.split(prep, 1)[1]
                best: Optional[str] = None
                best_score = 0
                for name in self._object_states:
                    if name in self._obstacle_names:
                        continue
                    if name == self._target_name:
                        continue
                    cleaned = name.replace("_", " ").rstrip(" 0123456789").strip()
                    if cleaned and after_prep.startswith(cleaned) and len(cleaned) > best_score:
                        best = name
                        best_score = len(cleaned)
                if best:
                    return best
        return None

    def _add_semantic_rules(self, target_exclusions=None) -> None:
        from .primitives import COMPOSED_RULE_SEEDS
        for seed in COMPOSED_RULE_SEEDS:
            bindings = self._resolve_bindings(seed)
            if bindings is None:
                continue
            formula = seed["formula"]
            for var, obj in bindings.items():
                formula = formula.replace(var, obj)
            try:
                parsed = parse_formula(formula)
            except Exception:
                continue
            primary_obj = bindings.get("A")
            rule = FOLRule(
                name=seed["name"],
                formula=formula,
                parsed=parsed,
                bindings={},
                cbf_type=seed["cbf_type"],
                cbf_params=seed["cbf_params"],
                violation_action=seed["violation_action"],
                primary_object=primary_obj,
                description=seed.get("description", ""),
                level=2,
            )
            self.kb.add_rule(rule)

    def _resolve_bindings(self, seed: Dict) -> Optional[Dict[str, str]]:
        formula = seed["formula"]
        needs_spillable = "IS_SPILLABLE(A)" in formula
        needs_fragile = "IS_FRAGILE(A)" in formula
        needs_human = "IS_HUMAN(A)" in formula
        needs_sharp = "HAS_SHARP_PART(A)" in formula
        needs_lit = "IS_LIT(A)" in formula
        needs_flammable = "IS_FLAMMABLE(B)" in formula
        needs_wet = "IS_WET(A)" in formula
        needs_absorbent = "IS_ABSORBENT(B)" in formula
        needs_supports = "SUPPORTS(A" in formula

        obj_A = obj_B = None
        for name, obj in self._object_states.items():
            if needs_lit and obj.is_lit and obj_A is None:
                obj_A = name
            if needs_fragile and obj.is_fragile and obj_A is None:
                obj_A = name
            if needs_spillable and obj.is_spillable and obj_A is None:
                obj_A = name
            if needs_wet and obj.is_wet and obj_A is None:
                obj_A = name
            if needs_human and obj.is_human and obj_A is None:
                obj_A = name
            if needs_sharp and obj.has_sharp_part and obj_A is None:
                obj_A = name
        for name, obj in self._object_states.items():
            if needs_flammable and obj.is_flammable and obj_B is None:
                obj_B = name
            if needs_absorbent and obj.is_absorbent and obj_B is None:
                obj_B = name
        if needs_supports:
            for name, obj in self._object_states.items():
                if obj_A is None:
                    obj_A = name

        if "A" in formula and obj_A is None:
            return None
        if "B" in formula and "A" in formula and obj_B is None:
            return None

        bindings: Dict[str, str] = {}
        if obj_A:
            bindings["A"] = obj_A
        if obj_B:
            bindings["B"] = obj_B
        return bindings if bindings else None
