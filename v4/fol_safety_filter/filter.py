"""
FOL Safety Filter v4 — VLM-Grounded Obstacle Detection + Predicate Composition

Changes vs v3:
  1. VLMObstacleGrounder: identifies obstacle from obs dict + optionally VLM,
     replacing the eval script's joint-name scan with use_vlm_obstacle flag.
  2. RuleComposer: generates FOL rules from grounded predicate dict
     (NEAR, IS_SPILLABLE→rotation-lock, IS_FRAGILE→wider radius, IS_HOT→zone).
  3. kb.py accepts external rule list — rules composed at episode start, not hard-coded.

All CBF geometry (carrying-phase detection, goal-tangential redirect) is
byte-for-byte identical to v3.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import of SemanticSafetyFilter
# ---------------------------------------------------------------------------

def _load_semantic_filter():
    _vlm_dir = os.path.join(os.path.dirname(__file__), "..", "vlm_pipeline")
    if _vlm_dir not in sys.path:
        sys.path.insert(0, _vlm_dir)
    from semantic_cbf_filter import SemanticSafetyFilter
    return SemanticSafetyFilter


# ---------------------------------------------------------------------------
# Imports from this package
# ---------------------------------------------------------------------------

from .primitives import (
    ObjectState,
    RobotState,
    VLM_GROUNDING_QUERIES,
    COMPOSED_RULE_SEEDS,
    PREDICATE_REGISTRY,
)
from .kb import FOLKnowledgeBase, FOLRule, ActiveConstraint, parse_formula
from .cbf_mapper import CBFMapper, MappedCBFSet
from .vlm_grounder import VLMObstacleGrounder
from .rule_composer import compose_rules


# ---------------------------------------------------------------------------
# VLM Grounder — physical-state binary queries (unchanged from v3)
# ---------------------------------------------------------------------------

class VLMGrounder:
    """Runs VLM binary queries to fill physical-state fields on ObjectState."""

    def __init__(self, use_vlm: bool = False, n_votes: int = 3):
        self.use_vlm = use_vlm
        self.n_votes = n_votes

    def ground(self, obs: Dict, object_states: Dict[str, ObjectState],
               rgb_image: Optional[np.ndarray] = None) -> None:
        if not self.use_vlm:
            for name, obj in object_states.items():
                self._heuristic_ground(name, obj)
            return

        for attr, query_template in VLM_GROUNDING_QUERIES.items():
            for name, obj in object_states.items():
                query = query_template.format(obj=name.replace("_", " "))
                try:
                    result = self._vlm_binary_query(query, rgb_image, votes=self.n_votes)
                    setattr(obj, attr, result)
                except Exception as e:
                    logger.debug(f"VLM query failed for {attr}({name}): {e}")
                    setattr(obj, attr, False)

    def _heuristic_ground(self, name: str, obj: ObjectState) -> None:
        n = name.lower().replace("_", " ")
        obj.is_fragile   = any(k in n for k in ["egg", "glass", "vase", "crystal"])
        obj.is_wet       = any(k in n for k in ["wet", "sponge", "water"])
        obj.is_lit       = any(k in n for k in ["candle lit", "torch", "flame"])
        obj.is_full      = any(k in n for k in ["bowl", "cup", "glass", "mug", "pot", "container"])
        obj.is_open      = any(k in n for k in ["bowl", "cup", "glass", "mug"])
        obj.is_flammable = any(k in n for k in ["napkin", "paper", "cloth", "fabric", "wood", "cardboard"])
        obj.is_absorbent = any(k in n for k in ["napkin", "paper", "cloth", "fabric", "sponge"])
        obj.is_human     = any(k in n for k in ["human", "hand", "arm", "person", "body"])
        obj.has_sharp_part = any(k in n for k in ["knife", "scissors", "blade", "sharp"])
        obj.is_heavy     = any(k in n for k in ["heavy", "moka", "iron", "cast"])
        obj.is_spillable = bool(obj.is_full and obj.is_open)

    def _vlm_binary_query(self, question: str, image: Optional[np.ndarray],
                           votes: int = 3) -> bool:
        import subprocess, json, tempfile
        results = []
        for _ in range(votes):
            try:
                result = subprocess.run(
                    ["conda", "run", "-n", "qwen", "python", "-c",
                     f"import json; from qwen_vlm_worker import answer_binary; "
                     f"print(answer_binary('{question}'))"],
                    capture_output=True, text=True, timeout=10,
                )
                ans = result.stdout.strip().lower()
                results.append("yes" in ans or "true" in ans)
            except Exception:
                results.append(False)
        return results.count(True) > votes // 2


# ---------------------------------------------------------------------------
# Main FOL Safety Filter v4
# ---------------------------------------------------------------------------

@dataclass
class FilterMetrics:
    timesteps: int = 0
    interventions: int = 0
    rule_counts: Dict[str, int] = field(default_factory=dict)


class FOLSafetyFilter:
    """
    Policy-agnostic FOL safety filter — v4.

    New vs v3:
      - VLMObstacleGrounder identifies obstacle from obs dict keys (+ optionally VLM)
      - RuleComposer generates FOL rules from grounded predicates
      - Rules include property-aware variants (IS_SPILLABLE, IS_FRAGILE, IS_HOT)
      - All v3 CBF geometry preserved unchanged
    """

    def __init__(
        self,
        use_vlm_grounding: bool = False,
        use_vlm_obstacle: bool = True,     # NEW: use VLM for obstacle detection (falls back to heuristic)
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
        self._alpha_default = alpha_default
        self._alpha_caution = alpha_caution

        self._object_states: Dict[str, ObjectState] = {}
        self._obstacle_name: Optional[str] = None
        self._target_name: Optional[str] = None
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
        obstacle_name: Optional[str] = None,  # GT from eval script (fallback)
        env=None,
        rgb_image: Optional[np.ndarray] = None,
    ) -> None:
        self.kb.clear()
        self._object_states = {}
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

        logger.info(f"[FOL-v4] Episode setup: {len(self._object_states)} objects: "
                    f"{list(self._object_states.keys())}")

        # Step 2: VLM obstacle detection (v4 addition)
        # Gather up to 2 camera views; eye_in_hand may equal agentview but still useful
        agentview = rgb_image if rgb_image is not None else obs.get("agentview_image")
        eye_in_hand = obs.get("robot0_eye_in_hand_image")
        rgb_images = [img for img in [agentview, eye_in_hand] if img is not None]
        candidates = list(self._object_states.keys())

        predicate_dict = self.obstacle_grounder.detect(rgb_images, task_description, candidates)
        vlm_obstacle = predicate_dict.get("obstacle_obs_key")

        if vlm_obstacle and vlm_obstacle in self._object_states:
            self._obstacle_name = vlm_obstacle
            logger.info(
                f"[FOL-v4] Obstacle detected ({predicate_dict['method']}): "
                f"{self._obstacle_name} | props={predicate_dict['physical_properties']}"
            )
        elif obstacle_name and obstacle_name in self._object_states:
            # Fall back to passed GT obstacle_name
            self._obstacle_name = obstacle_name
            logger.info(f"[FOL-v4] Obstacle fallback (GT): {self._obstacle_name}")
        else:
            self._obstacle_name = obstacle_name
            logger.warning(f"[FOL-v4] No obstacle found in scene. Passed: {obstacle_name}")

        # Step 3: Ground physical-state predicates (name heuristics or VLM)
        self.grounder.ground(obs, self._object_states, rgb_image)
        for name, obj in self._object_states.items():
            flags = {k: getattr(obj, k) for k in
                     ["is_fragile", "is_wet", "is_lit", "is_full", "is_open",
                      "is_flammable", "is_absorbent", "is_human", "has_sharp_part",
                      "is_heavy", "is_spillable"]
                     if getattr(obj, k)}
            if flags:
                logger.info(f"  {name}: {flags}")

        # Step 4: Build FOL rules via RuleComposer (v4 addition)
        self.kb.add_workspace_safety_rules()

        composed = compose_rules(predicate_dict, self._obstacle_name)
        for rule in composed:
            self.kb.add_rule(rule)

        # Level 2: semantic property rules for non-obstacle objects
        if self.fol_level >= 2:
            self._add_semantic_rules(
                target_exclusions={self._obstacle_name} if self._obstacle_name else None
            )

        logger.info(self.kb.summary())

        # Step 5: Initialize SemanticSafetyFilter
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

        self._target_name = self._extract_target_name(task_description)
        if self._target_name:
            logger.info(f"[FOL-v4] Target object: {self._target_name}")

        self._initialized = True
        logger.info(f"[FOL-v4] Episode ready: {len(self.kb.rules)} rules, "
                    f"obstacle={self._obstacle_name}, vlm={self.use_vlm_obstacle}")

    # ------------------------------------------------------------------ #
    # Per-timestep certification (identical to v3)
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
        u_safe = self._apply_cbf(u_cmd, state, cbf_set, arm_checkpoints)

        if np.linalg.norm(u_safe[:3] - u_cmd[:3]) > 1e-6:
            self.metrics.interventions += 1

        return u_safe

    def _get_arm_checkpoints(self, env) -> List[np.ndarray]:
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
    # Internal helpers (unchanged from v3)
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

        joint_pos = np.zeros(7)
        joint_vel = np.zeros(7)
        joint_lo = np.array([-2.9, -1.8, -2.9, -3.1, -2.9, -0.1, -2.9])
        joint_hi = np.array([ 2.9,  1.8,  2.9, -0.1,  2.9,  3.8,  2.9])
        ee_vel = np.zeros(3)

        return RobotState(
            ee_pos=_ee_pos,
            ee_quat=_ee_quat,
            ee_vel=ee_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_limits_lo=joint_lo,
            joint_limits_hi=joint_hi,
            gripper_width=gripper_w,
            objects=self._object_states,
        )

    def _extract_target_name(self, task_description: str) -> Optional[str]:
        if not task_description:
            return None
        desc_lower = task_description.lower()

        for prep in [" on the ", " in the ", " inside the ", " into the ", " onto the "]:
            if prep in desc_lower:
                after_prep = desc_lower.split(prep, 1)[1]
                best_recep: Optional[str] = None
                best_recep_score: int = 0
                for name in self._object_states:
                    if self._obstacle_name and name == self._obstacle_name:
                        continue
                    if "obstacle" in name.lower():
                        continue
                    cleaned = name.replace("_", " ").rstrip(" 0123456789").strip()
                    if cleaned and after_prep.startswith(cleaned) and len(cleaned) > best_recep_score:
                        best_recep = name
                        best_recep_score = len(cleaned)
                if best_recep:
                    logger.debug(f"[FOL] Goal receptacle target: {best_recep} (prep='{prep.strip()}')")
                    return best_recep

        best_match: Optional[str] = None
        best_score: int = 0
        for name in self._object_states:
            if self._obstacle_name and name == self._obstacle_name:
                continue
            if "obstacle" in name.lower():
                continue
            cleaned = name.replace("_", " ").rstrip(" 0123456789").strip()
            if cleaned and cleaned in desc_lower and len(cleaned) > best_score:
                best_match = name
                best_score = len(cleaned)
        return best_match

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
            except Exception as e:
                logger.debug(f"Could not parse composed rule {seed['name']}: {e}")
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
            logger.info(f"[FOL] Composed rule {rule.name}: {formula}")

    def _resolve_bindings(self, seed: Dict) -> Optional[Dict[str, str]]:
        formula = seed["formula"]

        needs_lit = "IS_LIT(A)" in formula
        needs_flammable = "IS_FLAMMABLE(B)" in formula
        needs_wet = "IS_WET(A)" in formula
        needs_absorbent = "IS_ABSORBENT(B)" in formula
        needs_spillable = "IS_SPILLABLE(A)" in formula
        needs_fragile = "IS_FRAGILE(A)" in formula
        needs_supports = "SUPPORTS(A" in formula
        needs_human = "IS_HUMAN(A)" in formula
        needs_sharp = "HAS_SHARP_PART(A)" in formula

        obj_A = obj_B = None

        for name, obj in self._object_states.items():
            if needs_lit and obj.is_lit:
                obj_A = name; break
        for name, obj in self._object_states.items():
            if needs_flammable and obj.is_flammable:
                obj_B = name; break
        for name, obj in self._object_states.items():
            if needs_wet and obj.is_wet and obj_A is None:
                obj_A = name; break
        for name, obj in self._object_states.items():
            if needs_absorbent and obj.is_absorbent and obj_B is None:
                obj_B = name; break
        for name, obj in self._object_states.items():
            if needs_spillable and obj.is_spillable and obj_A is None:
                obj_A = name; break
        for name, obj in self._object_states.items():
            if needs_fragile and obj.is_fragile and obj_A is None:
                obj_A = name; break
        for name, obj in self._object_states.items():
            if needs_human and obj.is_human and obj_A is None:
                obj_A = name; break
        for name, obj in self._object_states.items():
            if needs_sharp and obj.has_sharp_part and obj_A is None:
                obj_A = name; break

        if needs_supports and obj_A is None:
            for name in self._object_states:
                if "obstacle" not in name.lower():
                    obj_A = name; break

        needs_A = any([needs_lit, needs_wet, needs_spillable, needs_fragile,
                       needs_supports, needs_human, needs_sharp])
        needs_B = needs_flammable or needs_absorbent

        if needs_A and obj_A is None:
            return None
        if needs_B and obj_B is None:
            return None

        bindings = {}
        if obj_A:
            bindings["A"] = obj_A
        if obj_B:
            bindings["B"] = obj_B
        return bindings if bindings else None

    def _apply_cbf(self, u_cmd: np.ndarray, state: RobotState,
                    cbf_set: MappedCBFSet,
                    arm_checkpoints: Optional[List[np.ndarray]] = None) -> np.ndarray:
        u = u_cmd.copy()
        ee = state.ee_pos

        if self._obstacle_name and self._obstacle_name in state.objects:
            obj = state.objects[self._obstacle_name]
            obs_pos = obj.pos

            target_pos: Optional[np.ndarray] = None
            if self._target_name and self._target_name in state.objects:
                target_pos = state.objects[self._target_name].pos

            carrying = state.gripper_width < 0.025
            if carrying:
                eef_warning_r, eef_hard_r = 0.12, 0.06
            else:
                eef_warning_r, eef_hard_r = 0.20, 0.10

            self._apply_point_cbf(u, ee, obs_pos,
                                   warning_r=eef_warning_r, hard_r=eef_hard_r,
                                   push_hard=0.08, label="EEF",
                                   target_pos=target_pos, carrying=carrying)

            for cp_pos in (arm_checkpoints or []):
                self._apply_point_cbf(u, cp_pos, obs_pos,
                                       warning_r=0.15, hard_r=0.08,
                                       push_hard=0.06, label="ARM")

        if cbf_set.velocity_limit is not None or self.velocity_limit_default is not None:
            vlim = cbf_set.velocity_limit or self.velocity_limit_default
            speed = np.linalg.norm(u[:3])
            if speed > vlim:
                u[:3] *= vlim / speed

        if cbf_set.pose_constrained and cbf_set.angular_limit is not None:
            omega = np.linalg.norm(u[3:6])
            if omega > cbf_set.angular_limit:
                u[3:6] *= cbf_set.angular_limit / omega

        return u

    def _apply_point_cbf(
        self,
        u: np.ndarray,
        point: np.ndarray,
        obs_pos: np.ndarray,
        warning_r: float,
        hard_r: float,
        push_hard: float,
        label: str = "",
        target_pos: Optional[np.ndarray] = None,
        carrying: bool = False,
    ) -> None:
        diff = point - obs_pos
        dist = float(np.linalg.norm(diff))
        if dist <= 0:
            return

        point_next = point + u[:3]
        dist_next = float(np.linalg.norm(point_next - obs_pos))

        if dist >= warning_r and dist_next >= warning_r:
            return

        out_dir = diff / dist

        approach_comp = float(np.dot(u[:3], -out_dir))
        if approach_comp > 0:
            cancel_fraction = 1.0
            push_dir = out_dir

            if carrying:
                cancel_fraction = 0.15
                if target_pos is not None:
                    obs_to_target = float(np.linalg.norm(target_pos - obs_pos))
                    if obs_to_target < warning_r * 2.0:
                        target_diff = target_pos - point
                        target_dist = float(np.linalg.norm(target_diff))
                        if target_dist > 1e-6:
                            target_dir = target_diff / target_dist
                            speed = float(np.linalg.norm(u[:3]))
                            toward_target = float(np.dot(u[:3] / (speed + 1e-9), target_dir))
                            if toward_target > 0.5:
                                tangential = target_dir - np.dot(target_dir, out_dir) * out_dir
                                tang_norm = float(np.linalg.norm(tangential))
                                if tang_norm > 0.1:
                                    tangential /= tang_norm
                                    redirect = 0.6 * out_dir + 0.4 * tangential
                                    push_dir = redirect / float(np.linalg.norm(redirect))
                logger.debug(
                    f"[FOL-CBF] {label} carrying allowance: cancel={cancel_fraction:.2f} dist={dist:.3f}"
                )
            elif target_pos is not None:
                obs_to_target = float(np.linalg.norm(target_pos - obs_pos))
                if obs_to_target < warning_r * 2.0:
                    target_diff = target_pos - point
                    target_dist = float(np.linalg.norm(target_diff))
                    if target_dist > 1e-6:
                        target_dir = target_diff / target_dist
                        speed = float(np.linalg.norm(u[:3]))
                        toward_target = float(np.dot(u[:3] / (speed + 1e-9), target_dir))
                        if toward_target > 0.5:
                            cancel_fraction = 0.6
                            logger.debug(
                                f"[FOL-CBF] {label} last-mile allowance: "
                                f"obs_to_target={obs_to_target:.3f} toward={toward_target:.2f}"
                            )

            u[:3] += approach_comp * cancel_fraction * push_dir
            logger.debug(f"[FOL-CBF] {label} cancel {cancel_fraction:.2f}: dist={dist:.3f}")

        if dist < hard_r:
            push = (hard_r - dist) / hard_r * push_hard
            u[:3] += out_dir * push

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def report(self) -> Dict:
        m = self.metrics
        far = m.interventions / max(m.timesteps, 1)
        return {
            "timesteps": m.timesteps,
            "interventions": m.interventions,
            "filter_activation_rate": far,
            "rule_counts": m.rule_counts,
        }
