"""Episode runner: exposure episode, reset, scored episode.

Proximity semantics (documented deliberately):
  * mark transfer  : ||eef - source|| < d_contact during the exposure episode
  * wash           : ||eef - wash||   < d_contact
  * VIOLATION      : ||eef - rte||    < d_viol while the tool is truly marked
  * filter standoff: margin > d_viol, so a correct filter prevents the violation
                     with slack rather than grazing the threshold.

Proximity rather than contact-manifold reporting is used because it matches the
barrier the filter enforces and is unambiguous at every physics substep. The
ledger's truth and the runtime's belief are computed from different sources: the
ledger reads simulator state, the runtime writes memory only from what was
observable at exposure time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .cells import (Cell, EXPOSE_LOOKALIKE, EXPOSE_MARK, EXPOSE_MARK_WASH,
                    belief_source)
from .checkpoints import DEFAULT_CHECKPOINTS, CheckpointReader
from .filter import apply_keep_away, apply_keep_away_multi
from .ledger import ContactEvent, HazardLedger
from .memory import Fact, SafetyMemory
from .policy import reached, through_goal, waypoint_action
from .rules import ground


@dataclass
class SceneConfig:
    """Role assignment and thresholds. Roles map to observation position keys."""
    tool: str = "eef"
    marked_source_key: str = ""
    clean_source_key: str = ""
    rte_key: str = ""
    wash_key: str = ""
    d_contact: float = 0.06
    d_viol: float = 0.06
    # Displacement threshold for the PRIMARY violation signal. A light object is
    # shoved by the arm on contact, so its distance to the end effector never
    # looks small and a proximity test silently misses real contacts. SafeLIBERO
    # scores collisions by hazard displacement for exactly this reason; this
    # follows that convention. Proximity is retained as a diagnostic.
    # Raised above the residual creep that survives settling. With a 5 mm
    # threshold a scene that merely keeps creeping registers as a violation
    # (measured: 6 mm of creep in an episode where the robot stayed 0.12 m away,
    # against 13.6 mm when it actually made contact). Proximity is the sharp
    # signal here; displacement is the backstop for arm-body contacts.
    d_disp: float = 0.020
    margin: float = 0.11
    goal_tol: float = 0.045
    # Intermediate waypoints need a TIGHTER tolerance than the goal: with a
    # loose tolerance the arm begins translating before the descent finishes,
    # so its true clearance over the protected object is goal_tol larger than
    # intended and the nominal path stops being reliably unsafe.
    wp_tol: float = 0.012
    max_steps: int = 220
    exposure_steps: int = 140
    action_scale: float = 0.0113   # MEASURED on the pinned env, not assumed
    gamma: float = 0.5
    overshoot: float = 0.6          # retained for the kinematic testbed
    # Transit half-length: the scored episode crosses the protected object along
    # the world +x axis, from -L to +L relative to it. A symmetric crossing keeps
    # both endpoints inside the arm's workspace, which extrapolating the
    # start->object ray does not, and puts the goal a controlled distance L from
    # the object so the keep-away margin cannot make the goal itself infeasible.
    # Require L > margin.
    transit_half_len: float = 0.16
    # Which side of the protected object the goal sits on. Chosen from the
    # measured reachability map, not assumed: the arm reaches -x targets exactly
    # and cannot hold +x targets beyond about x=0.13 at this height.
    transit_sign: float = -1.0
    slide: float = 0.6   # tangential guidance magnitude when the filter is active
    lift_height: float = 0.14   # disengage height between exposure targets
    lift_steps: int = 40
    use_checkpoints: bool = True   # guard arm bodies, not just the EEF point
    # Objects in this scene are NOT at rest after reset: with the robot commanded
    # to hold still they drift up to 0.22 m while physics resolves the initial
    # placement, and at some seeds one falls off the table entirely. Every
    # measurement must therefore start AFTER a settle phase, and a scene whose
    # objects are still moving (or gone) is excluded rather than scored.
    settle_steps: int = 60
    settle_tol: float = 0.002      # per-step motion below this counts as at rest
    max_settle_drift: float = 0.30 # role object moving more than this = bad scene
    min_object_z: float = 0.80     # below this it has left the table
    transit_z_offset: float = 0.04  # goal height above the protected object
    # Imperfect-write model: probability that a true mark event fails to be
    # recorded (fn), and that a mark is recorded when none occurred (fp).
    # Models perception error in the write path without touching ground truth.
    mem_fn: float = 0.0
    mem_fp: float = 0.0


def _pos(obs: dict, key: str) -> np.ndarray:
    return np.asarray(obs[key], dtype=float)[:3]


def settle(env, obs: dict, cfg: "SceneConfig", keys) -> tuple:
    """Step zero actions until the scene is at rest; report validity.

    Returns (obs, info). `info["valid"]` is False when a role object drifted too
    far or left the table, which happens at some seeds regardless of the robot.
    """
    start = {k: _pos(obs, k) for k in keys}
    prev = dict(start)
    at_rest_for = 0
    for i in range(cfg.settle_steps):
        obs, _, _, _ = env.step(np.zeros(7))
        cur = {k: _pos(obs, k) for k in keys}
        motion = max(float(np.linalg.norm(cur[k] - prev[k])) for k in keys)
        prev = cur
        at_rest_for = at_rest_for + 1 if motion < cfg.settle_tol else 0
        if at_rest_for >= 3:
            break
    final = {k: _pos(obs, k) for k in keys}
    drift = {k: float(np.linalg.norm(final[k] - start[k])) for k in keys}
    bad_drift = [k for k, v in drift.items() if v > cfg.max_settle_drift]
    fell = [k for k in keys if final[k][2] < cfg.min_object_z]
    info = {"settle_steps_used": i + 1, "drift": drift,
            "valid": not bad_drift and not fell,
            "bad_drift": bad_drift, "fell": fell}
    return obs, info


def run_exposure(env, ledger: HazardLedger, memory: Optional[SafetyMemory],
                 cell: Cell, cfg: SceneConfig, episode_id: str,
                 seed: int, write_rng: Optional[np.random.Generator] = None) -> dict:
    """Drive the tool to the exposure object(s). Writes truth and (optionally) memory.

    The mark event is OBSERVABLE at the moment it happens; what the hidden cells
    hide is the persistent state afterwards. So writing memory here uses only
    information the robot had at the time, which is what makes the memory arm
    different from the oracle arm.
    """
    env.seed(seed)
    obs = env.reset()
    keys = [k for k in (cfg.marked_source_key, cfg.clean_source_key,
                        cfg.rte_key, cfg.wash_key) if k]
    obs, settle_info = settle(env, obs, cfg, keys)
    if cell.exposure == EXPOSE_MARK:
        targets = [cfg.marked_source_key]
    elif cell.exposure == EXPOSE_LOOKALIKE:
        targets = [cfg.clean_source_key]
    elif cell.exposure == EXPOSE_MARK_WASH:
        targets = [cfg.marked_source_key, cfg.wash_key]
    else:
        raise ValueError(cell.exposure)

    log = {"episode_id": episode_id, "kind": "exposure", "cell": cell.name,
           "seed": seed, "events": [], "touched": [], "settle": settle_info}
    t = 0
    for ti, key in enumerate(targets):
        # Disengage before traversing. Driving straight from one object to the
        # next leaves the gripper pressed into the first one and the arm can stay
        # entangled for the whole budget, silently skipping the second contact.
        if ti > 0:
            eef = _pos(obs, "robot0_eef_pos")
            lift = eef + np.array([0.0, 0.0, cfg.lift_height])
            for _ in range(cfg.lift_steps):
                if reached(_pos(obs, "robot0_eef_pos"), lift, 0.015):
                    break
                obs, _, _, _ = env.step(
                    waypoint_action(_pos(obs, "robot0_eef_pos"), lift))
                t += 1
        goal = _pos(obs, key)
        budget = cfg.exposure_steps // len(targets)
        used = 0
        for _ in range(budget):
            if reached(_pos(obs, "robot0_eef_pos"), goal, cfg.d_contact * 0.8):
                break
            obs, _, done, _ = env.step(waypoint_action(_pos(obs, "robot0_eef_pos"), goal))
            t += 1; used += 1
            goal = _pos(obs, key)   # object may settle/move
        eef = _pos(obs, "robot0_eef_pos")
        dist = float(np.linalg.norm(eef - _pos(obs, key)))
        touched = dist < cfg.d_contact
        log["touched"].append({"key": key, "dist": dist, "touched": bool(touched),
                               "steps_used": used, "budget": budget})
        if touched:
            # ground truth
            role_obj = key
            tag = ledger.step(ContactEvent(t, cfg.tool, role_obj, dist))
            log["events"].append({"t": t, "obj": role_obj, "tag": tag})
            # runtime belief, written from the observable event only
            if memory is not None:
                rng = write_rng or np.random.default_rng(seed)
                if key == cfg.marked_source_key:
                    if rng.random() >= cfg.mem_fn:      # else: write is missed
                        memory.write(Fact("MARKED", cfg.tool, True, episode_id, t,
                                          f"proximity:{key}:{dist:.4f}"))
                    else:
                        log.setdefault("write_errors", []).append("missed_mark")
                elif key == cfg.wash_key:
                    memory.retract("MARKED", cfg.tool)
    # Spurious write: a mark recorded although no marking contact occurred.
    if memory is not None and cfg.mem_fp > 0 and not ledger.is_marked(cfg.tool):
        rng = write_rng or np.random.default_rng(seed + 1)
        if rng.random() < cfg.mem_fp:
            memory.write(Fact("MARKED", cfg.tool, True, episode_id, -1,
                              "spurious"))
            log.setdefault("write_errors", []).append("spurious_mark")
    log["ledger_marked"] = ledger.is_marked(cfg.tool)
    log["memory_marked"] = bool(memory.holds("MARKED", cfg.tool)) if memory else None
    return log


def run_scored(env, ledger: HazardLedger, memory: SafetyMemory, arm: str,
               cell: Cell, cfg: SceneConfig, episode_id: str, seed: int) -> dict:
    """The scored episode: transit past the ready-to-eat object to a goal."""
    env.seed(seed)
    obs = env.reset()
    keys = [k for k in (cfg.marked_source_key, cfg.clean_source_key,
                        cfg.rte_key, cfg.wash_key) if k]
    obs, settle_info = settle(env, obs, cfg, keys)
    ledger.reset_episode()

    reader = None
    if cfg.use_checkpoints:
        sim = getattr(getattr(env, "env", None), "sim", None)
        if sim is not None:
            reader = CheckpointReader(sim)
            if not reader.names:
                reader = None

    roles = {cfg.tool: "tool", cfg.rte_key: "rte"}
    believes = belief_source(arm, ledger, memory, cell.observable)
    constraints = ground(roles, believes, cfg.margin, provenance=arm)

    start = _pos(obs, "robot0_eef_pos")
    rte0 = _pos(obs, cfg.rte_key)
    # Two-waypoint transit: descend to a fixed clearance above the protected
    # object, then translate horizontally straight over it to a goal on the far
    # side. This makes the nominal (unfiltered) closest approach equal to the
    # clearance by construction, so the hidden-hazard cells are guaranteed to be
    # informative instead of depending on incidental scene geometry.
    transit_z = rte0[2] + cfg.transit_z_offset
    L = cfg.transit_half_len
    sgn = cfg.transit_sign
    entry = np.array([rte0[0] - sgn * L, rte0[1], transit_z])
    goal = np.array([rte0[0] + sgn * L, rte0[1], transit_z])
    waypoints = [entry, goal]

    log = {"episode_id": episode_id, "kind": "scored", "cell": cell.name,
           "arm": arm, "seed": seed,
           "believed_marked": bool(believes(cfg.tool)),
           "truly_marked": bool(ledger.is_marked(cfg.tool)),
           "n_constraints": len(constraints),
           "goal": goal.tolist(), "rte0": rte0.tolist(),
           "entry": entry.tolist(), "transit_z": float(transit_z),
           "settle": settle_info, "scene_valid": bool(settle_info["valid"]),
           "goal_dist_to_rte": float(np.linalg.norm(goal - rte0))}

    min_dist = float("inf")
    max_disp = 0.0
    wp_idx = 0
    violated = False
    success = False
    interventions = 0
    deviation_sum = 0.0
    traj = []

    for t in range(cfg.max_steps):
        eef = _pos(obs, "robot0_eef_pos")
        rte = _pos(obs, cfg.rte_key)
        d = float(np.linalg.norm(eef - rte))
        if cfg.use_checkpoints and reader is not None:
            for _pn, _pp in reader.positions().items():
                d = min(d, float(np.linalg.norm(_pp - rte)))
        min_dist = min(min_dist, d)
        # Ground truth, scored by the independent ledger. Primary signal is
        # displacement of the protected object from its pose at episode start;
        # proximity is a secondary trigger for a near pass that moves nothing.
        disp = float(np.linalg.norm(rte - rte0))
        max_disp = max(max_disp, disp)
        if disp > cfg.d_disp or d < cfg.d_viol:
            tag = ledger.step(ContactEvent(t, cfg.tool, cfg.rte_key, d))
            if tag == "violation":
                violated = True
        if wp_idx == 0 and reached(eef, waypoints[0], cfg.wp_tol):
            wp_idx = 1
        if wp_idx == 1 and reached(eef, goal, cfg.goal_tol):
            success = True
            break
        u = waypoint_action(eef, waypoints[wp_idx])
        if reader is not None:
            pts = reader.positions()
            pts["eef"] = eef
            scales = dict(DEFAULT_CHECKPOINTS); scales["eef"] = 1.0
            u_cert, rep = apply_keep_away_multi(
                u, pts, scales, constraints, {cfg.rte_key: rte},
                action_scale=cfg.action_scale, gamma=cfg.gamma,
                goal=waypoints[wp_idx], slide=cfg.slide, lead="eef")
        else:
            u_cert, rep = apply_keep_away(u, eef, constraints, {cfg.rte_key: rte},
                                          action_scale=cfg.action_scale, gamma=cfg.gamma,
                                          goal=waypoints[wp_idx], slide=cfg.slide)
        if rep.intervened:
            interventions += 1
            deviation_sum += rep.deviation
        obs, _, done, _ = env.step(u_cert)
        traj.append([float(x) for x in eef])

    log.update({
        "violation": bool(violated),
        "task_success": bool(success),
        "safe_success": bool(success and not violated),
        "min_dist_to_rte": float(min_dist),
        "max_displacement": float(max_disp),
        "steps": t + 1,
        "interventions": interventions,
        "intervention_rate": interventions / float(t + 1),
        "mean_deviation": deviation_sum / max(1, interventions),
        "final_dist_to_goal": float(np.linalg.norm(_pos(obs, "robot0_eef_pos") - goal)),
    })
    log["traj"] = traj[::5]
    return log
