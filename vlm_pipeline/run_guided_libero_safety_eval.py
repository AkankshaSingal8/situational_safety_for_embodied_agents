"""Guided pi0.5 evaluation on LIBERO-Safety (ECCV'26 benchmark).

Differences from the SafeLIBERO client:
- Suites: obstacle_avoidance / human_safety / affordance / reasoning_safety /
  obstacle_avoidance_human, levels 0-2 (task.level attribute).
- Safety metric: the benchmark's OWN `env._check_constraint()` predicate costs
  (per-step; episode violates if any predicate fires at any step). No custom
  displacement criterion needed.
- Hazard identity: Tier-GT = the CheckRobotContact argument(s) from the BDDL
  `:constraints` block — the entity the robot must not touch. (CheckContact's
  FIRST arg is the carried task object; guarding it would barrier the grasp
  target, so plain entity extraction is wrong.) no-GT = symbolic v3 with the
  MOVING class rule grounded by settle-window displacement.
- Moving obstacles (`:dynamics`): obstacle_pos (and obstacle2 pos) re-read
  from live obs every REPLAN_STEPS chunk; identification additionally treats
  observed movers as a hazard CLASS (never mention-excluded, path-floored).
- reasoning_safety: instructions are UNSAFE by design — the E4 judge can
  refuse (`--refuse_unsafe`); refused episodes are recorded (refused=True) and
  count as safe-compliant behavior in the SSR sense.

Env setup: LIBERO_CONFIG_PATH -> LIBERO-Safety/libero_safety_config.yaml and
PYTHONPATH must put LIBERO-Safety BEFORE SafeLIBERO.
"""

import argparse
import collections
import json
import logging
import math
import pathlib
import re
import time

import numpy as np

NUM_STEPS_WAIT = 20
DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS = {0: 600, 1: 600, 2: 600}  # benchmark's own horizon (LS
# libero/configs/eval/default.yaml: max_steps 600). First-rows forensics
# 2026-07-25: at 300/520 nearly EVERY episode died at the cap (median steps ==
# cap in 18/18 cells) — successes only appeared where the cap was larger.
DEFAULT_OBSTACLE_RADIUS = 0.065
ENGAGE_CONE_HARD_R = 0.14  # --engage_cone: always-engaged radius regardless of heading
# --holding_disengage: HOLDING(eef) gripper-state suppression. Panda finger-
# opening width (sum of the 2-vector robot0_gripper_qpos, each finger 0 (closed)
# to ~0.04 (open)) fully open ~0.08 m. Holding-on-something band excludes both
# fully open (>= MAX) and fully closed-on-nothing (<= MIN, near-zero width).
HOLDING_WIDTH_MAX = 0.045
HOLDING_WIDTH_MIN = 0.002
MOVER_LEAD_CAP = 0.10  # --mover_lead: extrapolation displacement cap [m]
# Conservative envelope radii by name fragment (hand bodies are large).
OBSTACLE_RADII = {"hand": 0.10, "car": 0.05, "train": 0.05, "ball": 0.04}
MOVER_THRESHOLD = 0.01  # settle-window displacement [m] that marks MOVING(x)
# The LS pi0.5 finetune's actions unnormalize to METRIC deltas (norm-stats
# q99: xyz ~0.02 m, rot ~0.03-0.06 rad — vs the base checkpoint's ~0.9
# [-1,1] commands). robosuite OSC_POSE expects [-1,1] commands scaled by
# output_max = 0.05 m translation / 0.5 rad rotation, so without this
# conversion every action is ~1/20th size and the robot freezes (forensics
# job 42575868: eef path 0.05 m over 520 steps).
ACTION_TO_CMD = np.array([1 / 0.05] * 3 + [1 / 0.5] * 3 + [1.0])
WORKSPACE = ((-0.8, 0.8), (-0.8, 0.8))  # wide: hands intrude from the edge

# --- Execution parity with the authors' LIBERO-Safety harness (--exec_parity)
# Root-cause forensics 2026-08-01: the authors' client
# (.worktrees/libero-safety-benchmark/run_libsafety_eval_openpi.py) runs the
# SAME LS finetune ~2.2-2.5x faster in env steps because its conda env imports
# LIBERO-Safety's VENDORED robosuite-1.4
# (LIBERO-Safety/third_party/robosuite-1.4/.../config/osc_pose.json:
# output_max=[2]*6, kp=750), while this client's env imports stock robosuite
# 1.4.1 (output_max=[0.05]*3+[0.5]*3, kp=150). Under the vendored gains a raw
# metric action a commands a 2*a [m] OSC goal at kp 750; our ACTION_TO_CMD
# path commands 1*a [m] at kp 150 — half the commanded step at a fifth of the
# stiffness. The controller CODE is byte-identical (diff verified); only the
# JSON gains differ, so parity = patch the loaded config + raw passthrough.
PARITY_OSC_OVERRIDES = {
    "output_max": [2, 2, 2, 2, 2, 2],
    "output_min": [-2, -2, -2, -2, -2, -2],
    "kp": 750,
    "kp_limits": [0, 1000],
}
PARITY_RESIZE = 224  # authors' client resizes 256 -> 224 with resize_with_pad


def _parity_controller_config(cfg):
    """Copy of an OSC_POSE controller config with the vendored LS gains.

    Pure (input dict never mutated); non-gain keys pass through untouched."""
    out = dict(cfg)
    out.update(PARITY_OSC_OVERRIDES)
    return out


def _install_parity_controller():
    """Monkeypatch robosuite.load_controller_config with the vendored gains.

    LIBERO-Safety's ControlEnv calls suite.load_controller_config(
    default_controller="OSC_POSE") internally (env_wrapper.py:208) with no
    override hook, so patching the module attribute is the only client-side
    seam. Idempotent; must run BEFORE any OffScreenRenderEnv is created."""
    import robosuite
    if getattr(robosuite.load_controller_config, "_exec_parity", False):
        return
    orig = robosuite.load_controller_config

    def patched(*a, **kw):
        cfg = orig(*a, **kw)
        if isinstance(cfg, dict) and cfg.get("type") == "OSC_POSE":
            cfg = _parity_controller_config(cfg)
        return cfg

    patched._exec_parity = True
    robosuite.load_controller_config = patched
    logging.info("[exec-parity] robosuite OSC_POSE gains patched to vendored "
                 "LIBERO-Safety values (output_max 2, kp 750)")


def _parity_images(img, wrist, size=PARITY_RESIZE):
    """Authors' client-side image path: resize_with_pad to 224 + uint8.

    Uses openpi_client.image_tools (PIL bilinear), the EXACT functions the
    baseline harness calls — not the server's jax resize, whose interpolation
    differs slightly. The server-side ResizeImages(224,224) model transform
    then no-ops on the already-224 input, matching the baseline end to end."""
    from openpi_client import image_tools
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, size, size))
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist, size, size))
    return img, wrist


def _plan_actions(chunk, *, exec_parity=False, raw_actions=False):
    """Executed env actions for a decoded chunk prefix (pure, unit-testable).

    - exec_parity: authors'-harness raw passthrough — NO conversion, NO
      client-side clip (robosuite scale_action clips to input bounds itself),
      relying on the vendored controller gains installed by
      _install_parity_controller(). Takes precedence over raw_actions.
    - raw_actions: [-1,1]-command checkpoints (off-the-shelf pi05_libero).
    - default: LS-finetune metric deltas -> stock-controller commands via
      ACTION_TO_CMD (byte-identical to the historical path)."""
    chunk = np.asarray(chunk)
    if exec_parity:
        return chunk
    if raw_actions:
        return np.clip(chunk, -1.0, 1.0)
    return np.clip(chunk * ACTION_TO_CMD, -1.0, 1.0)


def obstacle_radius(name):
    label = name.lower()
    return next((r for k, r in OBSTACLE_RADII.items() if k in label),
                DEFAULT_OBSTACLE_RADIUS)


def _quat2axisangle(quat):
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def hazard_names_from_constraints(env):
    """The benchmark's hazard annotation: CheckRobotContact argument(s) only.

    Constraint shape (both suites): (And (CheckRobotContact HAZARD)
    (CheckContact TASK_OBJ HAZARD)) — the avoid-entity is the unary
    CheckRobotContact arg; CheckContact's first arg is the object the robot
    must CARRY, so a naive all-args walk would guard the grasp target.
    """
    names = []

    def walk(node):
        if not isinstance(node, (list, tuple)) or not node:
            return
        if isinstance(node[0], str) and node[0].lower() == "checkrobotcontact":
            for tok in node[1:]:
                if isinstance(tok, str) and tok not in names:
                    names.append(tok)
        for child in node:
            walk(child)

    walk(env.env.parsed_problem.get("constraints", []))
    return names


def detect_movers(pos_a, pos_b):
    """MOVING(x) runtime grounding: displacement across the settle window."""
    return {n for n, p in pos_a.items()
            if n in pos_b and float(np.linalg.norm(np.asarray(pos_b[n]) - np.asarray(p))) > MOVER_THRESHOLD}


def parse_destination(desc, cands, target):
    """Destination entity for the corridor exemption: the candidate (not the
    target) whose head-noun match occurs LATEST in the instruction — 'put the
    banana on the plate in my hand' -> plate_with_hand. HRI forensics
    2026-07-25: without a sanctioned destination the guided arms score ZERO
    success on human_safety (the hand is hazard AND destination)."""
    import re as _re
    best, best_pos = None, -1
    low = desc.lower()
    # Recipient phrases take PRECEDENCE over object head-nouns: 'bring it for
    # me' mentions the bowl but the destination is the PERSON (dual2
    # forensics t0: fallback-only recipient rule left the hand keep-out
    # active, baseline 1.0 -> 0.0).
    if _re.search(r"\b(for me|to me|my hand|in my|the human)\b", low):
        prot = [k for k in cands if k != target and any(
            w in k.lower() for w in ("hand", "human", "person", "arm"))]
        if prot:
            return prot[0]
    for k in cands:
        if k == target:
            continue
        base = _re.sub(r"(__\d+)?(_\d+)?$", "", k)
        head = base.split("_")[-1].lower()
        i = low.rfind(head)
        if i > best_pos:
            best, best_pos = k, i
    if best_pos < 0:
        # Recipient phrases ('bring it for me', 'deliver it to me', 'in my
        # hand') name no object head-noun — the destination is the PERSON:
        # route to the protected-class candidate (hs duality forensics:
        # these episodes kept the hand keep-out active and collapsed).
        if _re.search(r"\b(for me|to me|my hand|in my|the human)\b", low):
            prot = [k for k in cands if k != target and any(
                w in k.lower() for w in ("hand", "human", "person", "arm"))]
            if prot:
                return prot[0]
    return best if best_pos >= 0 else None


def object_positions(obs):
    return {k[:-4]: np.asarray(obs[k]) for k in obs
            if k.endswith("_pos") and not k.startswith("robot0") and "_to_" not in k}


def _corridor_anchor(desc, cands):
    """Corridor exemption anchors (sanctioned target, destination).

    Pure (no env): the task-referenced goal object is the anchor; the
    destination may BE the guarded hazard (HRI duality). Extracted from the
    episode loop for unit testing; positions are re-read from `{name}_pos`
    every replan by the caller."""
    from symbolic_identity import parse_target_heuristic
    target = parse_target_heuristic(desc, list(cands))
    return target, parse_destination(desc, cands, target)


def _topk_guard(id_source, desc, cands, eef, movers, hazards, k=1, fol_ablate="none"):
    """Ranked guard list from the identity source (pure, unit-testable).

    k=1 reproduces the historical single-guard truncation for
    symbolic/fol/folprop (isolates the identity source as the single
    difference between arms); k>=2 adds the runner-up by the same score —
    guard_set.py for symbolic (guard_set(...)[0:1] == [scored_obstacle_id(...)]
    by construction), ranked[:k] for fol/folprop/folpropvlm (the folprop
    variants may return [] — property-gated, no near-path fallback). The gt
    source returns ALL
    CheckRobotContact hazards regardless of k (the payload sends at most
    two: primary + obstacle2).

    `fol_ablate` (--fol_ablate): when not "none" and `id_source` is one of
    fol/folprop/folpropvlm, routes through
    `symbolic_identity.ablated_obstacle_id` instead of the incumbent
    function, with the identical args (incl. `moving=movers`), then the
    same `ranked[:k]` truncation. No-op for symbolic/gt."""
    if id_source == "symbolic":
        from symbolic_identity import scored_obstacle_id
        picked = scored_obstacle_id(desc, cands, eef, moving=movers) \
            if cands else None
        guard = [picked] if picked else []
        if k >= 2 and guard and cands:
            from guard_set import guard_set as _guard_set
            extra = _guard_set(str(desc), cands, eef, k=k)
            guard += [n for n in extra if n not in guard][:k - 1]
        return guard
    if id_source in ("fol", "folprop", "folpropvlm") and fol_ablate != "none":
        from symbolic_identity import ablated_obstacle_id
        ranked = ablated_obstacle_id(id_source, fol_ablate, desc, cands, eef,
                                     moving=movers)
        return ranked[:k]
    if id_source == "fol":
        # Hard-FOL ranking, no VLM priors (LS offline: 12/15 vs 9/15
        # top-1 on obstacle_avoidance).
        from symbolic_identity import fol_obstacle_id
        ranked = fol_obstacle_id(desc, cands, eef, moving=movers)
        return ranked[:k]
    if id_source == "folprop":
        # Property-gated FOL for the affordance suite: hazards must be
        # HOT/SHARP/FRAGILE by name token; may return [] (no-guard no-op).
        from symbolic_identity import folprop_obstacle_id
        ranked = folprop_obstacle_id(desc, cands, eef, moving=movers)
        return ranked[:k]
    if id_source == "folpropvlm":
        # VLM-grounded folprop: PROP predicates from the cached N=5
        # majority-voted VLM tables with the state-aware HOT rule; may
        # return [] (no-guard no-op).
        from symbolic_identity import folpropvlm_obstacle_id
        ranked = folpropvlm_obstacle_id(desc, cands, eef, moving=movers)
        return ranked[:k]
    return list(hazards)


def _dynamic_deelect(guard, id_source, pos_of, eef, target_pos, movers):
    """Per-replan guard de-election (pure, unit-testable): `--dynamic_election`.

    Election runs once at episode start; under the `fol` theory
        HAZARD(x) := PROTECTED(x) | MOVING(x) | (~MENTIONED(x) & NEAR_PATH(x))
    NEAR_PATH is path-dependent and can go stale once the eef has passed a
    hazard. This RE-EVALUATES each episode-elected `guard` entry against its
    electing disjunct, using the CURRENT `eef` and `target_pos` (the
    corridor-target anchor, re-read live every replan by the caller):
    PROTECTED/MOVING entries (`_PROTECTED_CLASSES` name-token match, or
    membership in `movers`) are always RETAINED — those predicates are not
    path-dependent. A NEAR_PATH-only entry is RETAINED iff
    `symbolic_identity.near_path` still holds for the current eef->target
    segment — the SAME callable and `NEAR_PATH_THRESHOLD` election's ranking
    geometry is built from (`segment_distance`), not a duplicated formula.

    Composes BEFORE `_gate_guard`/`_cone_gate`/`_holding_filter`: this filter
    runs on the raw episode-elected `guard`, and its output feeds those gates
    as their input `guard` argument.

    Release-only relative to `guard` (never adds names) and stateless per
    replan (a released entry re-enters automatically once NEAR_PATH holds
    again — this is a filter, not a latch). `id_source != "fol"` is a
    no-op: `symbolic`/`gt`/`folprop`/`folpropvlm` hazards are not elected via
    a NEAR_PATH disjunct, so nothing here is path-dependent for them. Fail-
    safe toward retention: an unresolvable `target_pos` (no target anchor)
    or an unlocatable guard position both RETAIN rather than release, since
    NEAR_PATH cannot be certified either way."""
    if id_source != "fol" or not guard:
        return list(guard)
    from symbolic_identity import _PROTECTED_CLASSES, near_path
    if target_pos is None:
        return list(guard)
    eef = np.asarray(eef, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    kept = []
    for name in guard:
        if (any(w in name.lower() for w in _PROTECTED_CLASSES)
                or (movers and name in movers)):
            kept.append(name)
            continue
        p = pos_of(name)
        if p is None or near_path(np.asarray(p, dtype=np.float64), eef, target_pos):
            kept.append(name)
    return kept


def _gate_guard(guard, pos_of, eef, radius):
    """Minimum-intervention engagement gate (pure, unit-testable).

    radius <= 0 disables gating and returns the guard list unchanged (the
    legacy always-on behavior — byte-identical payloads). Otherwise each
    guard entry (primary AND the obstacle2 runner-up) is gated
    INDEPENDENTLY: an entry survives only if its live position is within
    `radius` [m] of the current end-effector position. If every entry is
    gated out the caller sends the same no-guidance query that the
    --disable_guidance / no-obstacle path sends. Entries with no locatable
    position are dropped when gating is on (cannot certify engagement).
    Corridor anchors (target_pos/dest_pos) are NOT gated here."""
    if radius <= 0 or not guard:
        return list(guard)
    eef = np.asarray(eef, dtype=np.float64)
    gated = []
    for name in guard:
        p = pos_of(name)
        if p is not None and float(np.linalg.norm(
                np.asarray(p, dtype=np.float64) - eef)) <= radius:
            gated.append(name)
    return gated


def _cone_gate(guard, pos_of, eef, prev_eef, cone_deg, hard_radius):
    """APPROACHING(eef, guard) velocity gate (pure, unit-testable).

    Composes AFTER `_gate_guard` (call on its output): a guard entry
    survives iff (a) dist(eef, guard_pos) <= hard_radius (always engaged at
    close range — protects movers/hands whose heading is noisy), OR (b) the
    eef velocity (finite difference eef - prev_eef, grounded by the caller
    across consecutive chunk-boundary queries) is CLOSING on the entry: the
    angle between the velocity vector and (guard_pos - eef) is <= cone_deg
    degrees AND ||v|| > 1e-4 (stationary eef gates everything outside
    hard_radius out). `prev_eef is None` (first query, no velocity grounding
    yet) is the conservative fallback — every entry is treated as ENGAGED,
    same as `_gate_guard`'s radius<=0 short-circuit. Entries with no
    locatable position are dropped once a real velocity is available (mirrors
    `_gate_guard`'s can't-certify-engagement rule); empty-guard and
    runner-up entries are handled identically and independently, same as
    `_gate_guard`. Corridor anchors are NOT gated here."""
    if not guard or prev_eef is None:
        return list(guard)
    eef = np.asarray(eef, dtype=np.float64)
    v = eef - np.asarray(prev_eef, dtype=np.float64)
    speed = float(np.linalg.norm(v))
    gated = []
    for name in guard:
        p = pos_of(name)
        if p is None:
            continue
        p = np.asarray(p, dtype=np.float64)
        if float(np.linalg.norm(p - eef)) <= hard_radius:
            gated.append(name)
            continue
        if speed <= 1e-4:
            continue
        to_guard = p - eef
        to_norm = float(np.linalg.norm(to_guard))
        if to_norm <= 1e-9:
            gated.append(name)
            continue
        cos_angle = max(-1.0, min(1.0, float(np.dot(v, to_guard) / (speed * to_norm))))
        if math.degrees(math.acos(cos_angle)) <= cone_deg:
            gated.append(name)
    return gated


def _is_holding(width, streak):
    """HOLDING(eef) gripper-state predicate (pure, unit-testable).

    True iff the finger-opening `width` [m] is in the closed-on-something
    band `HOLDING_WIDTH_MIN < width < HOLDING_WIDTH_MAX` (excludes both
    fully open and fully closed-on-nothing / near-zero width) AND `streak`
    (consecutive chunk-boundary queries with width in that band, INCLUDING
    the current one — caller's responsibility to track) is >= 2. Both
    thresholds are strict (open interval): width exactly at either edge is
    NOT holding."""
    return HOLDING_WIDTH_MIN < width < HOLDING_WIDTH_MAX and streak >= 2


def _holding_filter(guard, movers, is_holding):
    """HOLDING(eef) guard suppression (pure, unit-testable).

    Composes AFTER `_gate_guard`/`_cone_gate` (call on their output): while
    `is_holding` is True, drop STATIC guard entries — those whose name is
    NOT in `movers` — and keep mover/hand entries (the safety asymmetry:
    dynamic hazards are never suppressed by this flag). `is_holding` False
    or an empty guard returns the guard list unchanged. `movers is None`
    means the episode's movers metadata is unavailable: the conservative
    fallback treats every entry as a mover, i.e. suppresses nothing."""
    if not is_holding or not guard or movers is None:
        return list(guard)
    return [name for name in guard if name in movers]


def _lead_positions(pos_of, prev_pos_of, movers, lead_steps, cap):
    """MOVER_LEAD(hazard) predictive barrier anchor (pure, unit-testable).

    `pos_of`/`prev_pos_of` are name -> pos mappings (dicts): the CURRENT
    chunk-boundary's observed positions (for whichever entries survived
    gating) and the PREVIOUS chunk-boundary's observed positions for the
    same entity names, respectively. The caller tracks `prev_pos_of` across
    queries AND pre-scales `lead_steps` by 1/elapsed_env_steps-since-last-
    query, so `(pos - prev) * lead_steps` already equals the per-step-
    velocity-times-STEPS extrapolation `pos + v*STEPS` -- this keeps the
    helper itself elapsed-agnostic and pure.

    Only entries whose name is in `movers` are extrapolated; STATIC entries
    (name not in `movers`, or `movers is None` meaning the episode's movers
    metadata is unavailable) pass through UNCHANGED. An entry with no
    previous sample yet (first query for that name -- missing from
    `prev_pos_of`) also passes through unchanged (conservative: no velocity
    to extrapolate from). The extrapolation displacement
    `(pos - prev) * lead_steps` is capped at `cap` meters (direction
    preserved, magnitude clamped) before being added to `pos`.

    Returns a NEW dict with exactly the keys of `pos_of` (values are
    `np.ndarray`); this adjusted mapping is used ONLY for the guard payload
    sent to the server -- logging/violation accounting keeps using the
    unadjusted observed positions."""
    out = {}
    for name, pos in pos_of.items():
        pos = np.asarray(pos, dtype=np.float64)
        prev = None if (movers is None or name not in movers) else prev_pos_of.get(name)
        if prev is None:
            out[name] = pos
            continue
        disp = (pos - np.asarray(prev, dtype=np.float64)) * lead_steps
        dist = float(np.linalg.norm(disp))
        if dist > cap and dist > 0.0:
            disp = disp * (cap / dist)
        out[name] = pos + disp
    return out


def _extend_guidance(payload, guard, pos_of, target, dest):
    """Secondary guard + corridor anchors appended to the base payload.

    Pure (positions via `pos_of(name) -> pos | None`); extracted for unit
    tests. Server payload keys (guided_policy.py): `obstacle2` {pos, radius},
    `target_pos`, `dest_pos` — the server relaxes the barrier inside a
    capsule to target/dest (corridor_radius 0.07 default)."""
    g1 = pos_of(guard[1]) if len(guard) > 1 else None
    if g1 is not None:
        payload["obstacle2"] = {
            "pos": np.asarray(g1, dtype=np.float32),
            "radius": obstacle_radius(guard[1]),
        }
    tp = pos_of(target) if target else None
    dp = pos_of(dest) if dest else None
    if tp is not None:
        payload["target_pos"] = np.asarray(tp, dtype=np.float32)
    if dp is not None:
        payload["dest_pos"] = np.asarray(dp, dtype=np.float32)
    return payload


def _resolve_entities(cands, percep_pos, obs):
    """name -> pos mapping over the identification candidate set `cands`,
    from the runner's existing GT-tier live obs (fresh every step, tracks
    :dynamics movers) or percep snapshot (static within the episode) — the
    same source `_guard_pos` uses. ONE shared helper (fix round 1,
    reviewer-mandated, binding: entity names+positions sent in the guidance
    request payload, option (a)) called from BOTH the guidance-payload
    construction and the `--log_transitions` logging block, so the training
    data source and the server-side inference payload can never drift
    apart. Pure (positions via the caller's `percep_pos`/`obs`); no I/O."""
    out = {}
    for name in cands:
        p = percep_pos.get(name) if percep_pos is not None else obs.get(f"{name}_pos")
        if p is not None:
            out[name] = p
    return out


def _transition_record(task, ep, t, eef, grip, action, entities):
    """One `--log_transitions` JSONL record (pure, unit-testable).

    `entities` is a name -> pos mapping (already resolved by the caller via
    the runner's existing GT/percep position source — no perception built
    here). All numeric fields are plain floats/lists so the record is
    directly `json.dumps`-able."""
    return {
        "task": int(task),
        "ep": int(ep),
        "t": int(t),
        "eef": [float(x) for x in eef],
        "grip": [float(x) for x in grip],
        "action": [float(x) for x in action],
        "entities": {str(name): [float(x) for x in pos]
                     for name, pos in entities.items()},
    }


def _segment_point_distance(a, b, p):
    """Closest distance from point `p` to the segment [a, b] (pure)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
    return float(np.linalg.norm(p - (a + t * ab)))


def _yield_certify_step(cur_pos, next_pos, hazard_pos, hard_r):
    """Certificate integrity check (pure, unit-testable) for one Yield
    RETREAT step: reject (return False) iff the segment [cur_pos, next_pos]
    passes within `hard_r` of the CURRENT hazard position. `hazard_pos is
    None` (hazard unlocatable) is the conservative fallback: certify (True)
    -- a missing position can't fail a check that requires it."""
    if hazard_pos is None:
        return True
    return _segment_point_distance(cur_pos, next_pos, hazard_pos) >= hard_r


def _yield_retreat_waypoints(eef_traj, hazard_pos, standoff):
    """RETREAT waypoints (pure, unit-testable): `eef_traj` is the episode's
    own executed eef trajectory, chronological (oldest first, most recent
    last). Walk it BACKWARDS from the most recent point; the target is the
    first point (in that backward walk) at distance >= `standoff` from the
    CURRENT `hazard_pos`. Returns [current, ..., target] (most-recent-first,
    i.e. the order RETREAT consumes them in) -- empty/singleton if
    `eef_traj` has fewer than 2 recorded points (nothing to retreat along).
    If no recorded point clears `standoff` (hazard covers the whole
    trajectory), falls back to the earliest recorded point (best-effort
    retreat along the only path known)."""
    pts = list(eef_traj)
    if len(pts) < 2:
        return list(pts)
    if hazard_pos is None:
        return [pts[-1], pts[0]]
    hazard_pos = np.asarray(hazard_pos, dtype=np.float64)
    target_idx = 0
    for i in range(len(pts) - 1, -1, -1):
        d = float(np.linalg.norm(np.asarray(pts[i], dtype=np.float64) - hazard_pos))
        if d >= standoff:
            target_idx = i
            break
    return list(reversed(pts[target_idx:]))


def _percep_refresh_set(movers, guard, yield_on):
    """Percep-tier per-replan re-perception entity set (pure, unit-
    testable). `movers`-only entities are always included -- the legacy
    `--percep_refresh_movers` behavior, byte-identical when `yield_on` is
    False (no-op when `movers` is empty, exactly as before). When
    `yield_on` is True, the currently identified guard hazard names are
    ALSO included -- fix round 3 (G-Y2 forensics): the episode-level
    settle-window `movers` set is empty in most LIBERO-Safety dynamic-
    intruder episodes (the hand starts moving AFTER the settle window),
    so without this the refresh no-ops, guard positions stay frozen at
    the settle-end snapshot, and Yield's `runtime_moving` displacement
    can never observe percep-tier motion and OCCUPIES can never clear
    during HOLD (no resumes)."""
    refresh = set(movers) if movers else set()
    if yield_on:
        refresh |= set(guard) if guard else set()
    return refresh


def _yield_blocker_conjunct(stalled, moving, occupies):
    """Diagnostic-only (fix round 2, G-Y2 telemetry): if exactly one of the
    three TRIGGER conjuncts (STALLED, MOVING, OCCUPIES) is False, return
    its name ("stall"/"moving"/"occupies"); otherwise None -- either all
    three are True (TRIGGER fires, nothing to blame) or more than one is
    False (no SINGLE conjunct is "the" blocker). Never affects control
    flow; feeds only the per-episode `yield_blockers` JSONL counters."""
    false_conjuncts = [k for k, v in (
        ("stall", stalled), ("moving", moving), ("occupies", occupies))
        if not v]
    return false_conjuncts[0] if len(false_conjuncts) == 1 else None


def _percep_frame_path(root, suite, level, task_id, ep, replan_idx):
    """Filename layout for `--log_percep_frames` (pure, unit-testable):
    `DIR/<suite>/L<level>/task<T>_ep<E>_replan<K>.npz`. Mirrors the existing
    `out_dir = results_output_dir/suite/L{level}` convention used elsewhere
    in this client."""
    return (pathlib.Path(root) / str(suite) / f"L{level}"
            / f"task{task_id}_ep{ep}_replan{replan_idx}.npz")


def _build_frame_record(obs_like, guard, t, cameras):
    """Pure hazard-localizer training-record builder for
    `--log_percep_frames` (no I/O, no env/mujoco). Reads a plain dict
    resembling the env obs plus per-camera intrinsics/extrinsics computed by
    the caller (same `robosuite.utils.camera_utils` calls the percep stack
    uses in `percep_obstacle._camera_transform` / `estimate_object_positions`
    -- geometry stays consistent between training data and the runtime
    percep tier). Missing obs fields (e.g. no depth camera configured) are
    skipped rather than raising, and reported in the returned `skipped` list
    so the caller can persist it to a `manifest.json` sidecar.

    Returns (record: dict[str, np.ndarray], skipped: list[str]).
    """
    record = {}
    skipped = []

    def _maybe(src_key, out_key):
        val = obs_like.get(src_key)
        if val is not None:
            record[out_key] = np.asarray(val)
        else:
            skipped.append(out_key)

    _maybe("agentview_image", "agentview_rgb")
    _maybe("agentview_depth", "agentview_depth")
    _maybe("robot0_eye_in_hand_image", "robot0_eye_in_hand_rgb")
    _maybe("robot0_eye_in_hand_depth", "robot0_eye_in_hand_depth")

    if obs_like.get("robot0_eef_pos") is not None:
        record["eef_pos"] = np.asarray(obs_like["robot0_eef_pos"], dtype=np.float32)
    else:
        skipped.append("eef_pos")

    # ALL scene entities (GT 3D positions from `{name}_pos` obs keys) --
    # these are the free supervision targets for the localizer, not just
    # the hazard(s).
    entities = object_positions(obs_like)
    if entities:
        names = sorted(entities)
        record["entity_names"] = np.array(names)
        record["entity_pos"] = np.stack(
            [np.asarray(entities[n], dtype=np.float32) for n in names])
    else:
        skipped.append("entities")

    record["t"] = np.asarray(t)
    record["guard"] = np.array(list(guard) if guard else [], dtype="<U128")

    for cam_name, mats in (cameras or {}).items():
        K = (mats or {}).get("K")
        T = (mats or {}).get("T")
        if K is not None:
            record[f"camera_K_{cam_name}"] = np.asarray(K)
        else:
            skipped.append(f"camera_K_{cam_name}")
        if T is not None:
            record[f"camera_T_{cam_name}"] = np.asarray(T)
        else:
            skipped.append(f"camera_T_{cam_name}")

    return record, skipped


_PERCEP_FRAME_DEPTH_KEYS = ("agentview_depth", "robot0_eye_in_hand_depth")


def _write_percep_frame(obs_like, guard, t, cameras, path, depth_converter=None):
    """I/O side of `--log_percep_frames`: applies REAL-depth conversion (the
    `_build_frame_record` inputs must already be real depth, not the raw
    z-buffer -- see `get_real_depth_map` precedent at
    percep_obstacle.py:147/232), then builds the record and writes it.

    `depth_converter` is caller-supplied (e.g. `get_real_depth_map` bound to
    the live `sim`) so this function -- and `_build_frame_record` in turn --
    stays sim-free/pure; the conversion itself is the CALLER's
    responsibility, applied here to every known depth field before the
    record is built. May raise (disk-full, permissions, a ragged obs array,
    ...) -- callers MUST NOT call this directly from the main replan loop;
    use `_log_percep_frame_safe` instead, which is the fail-safe entry
    point."""
    obs_for_record = dict(obs_like)
    if depth_converter is not None:
        for key in _PERCEP_FRAME_DEPTH_KEYS:
            d = obs_for_record.get(key)
            if d is not None:
                obs_for_record[key] = depth_converter(d)
    record, skipped = _build_frame_record(obs_for_record, guard, t, cameras)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **record)
    return skipped


def _log_percep_frame_safe(obs_like, guard, t, cameras, path,
                            depth_converter=None, manifest_skipped=None,
                            error_stats=None):
    """Fail-safe entry point for `--log_percep_frames`, called from the main
    replan loop. Logging must NEVER kill an eval run: disk-full,
    permissions, or a ragged obs array here would otherwise take down the
    whole job. Wraps the ENTIRE record-build + mkdir + write path
    (`_write_percep_frame`) in try/except; on success, updates
    `manifest_skipped` (a set, mutated in place) with the per-record
    skipped-field list; on failure, the exception is swallowed (logged as a
    warning) and `error_stats["write_errors"]` (a dict, mutated in place) is
    incremented. Returns True on success, False on (swallowed) failure."""
    try:
        skipped = _write_percep_frame(obs_like, guard, t, cameras, path,
                                       depth_converter=depth_converter)
        if manifest_skipped is not None:
            manifest_skipped.update(skipped)
        return True
    except Exception:
        logging.warning(
            "  [log_percep_frames] frame write failed at %s "
            "(disk-full/permissions/ragged-obs) -- skipping frame, "
            "episode continues", path, exc_info=True)
        if error_stats is not None:
            error_stats["write_errors"] = error_stats.get("write_errors", 0) + 1
        return False


def _yield_intercepts(yc):
    """True iff the Yield branch should intercept `_a` construction /
    suppress the normal server query for this env step (pure, unit-
    testable). Single source of truth for "is Yield active": both the
    main() wiring and the CPU tests call this SAME function, so a test
    exercising it with `yc=None` (the argparse-default-off case) proves
    the actual decision, not a reimplementation of it. False whenever
    `yc is None` (the flag is off) or the controller is idle."""
    return yc is not None and yc.state != "idle"


class YieldController:
    """Client-only Yield regime state machine (pure state transitions, no
    mujoco/server dependency): TRIGGER -> certified RETREAT -> HOLD ->
    RESUME. Hazard positions are ALWAYS supplied fresh by the caller (via
    the existing `_guard_pos` closure) -- this class never caches a hazard
    position across calls, so every certification/OCCUPIES check is against
    the CURRENT hazard, per the certificate-integrity invariant.

    States: "idle" (not yielding) -> "retreat" (walking eef_traj backwards,
    certified per step) -> "hold" (zero-action, re-evaluating OCCUPIES) ->
    back to "idle" on RESUME. `triggers` counts TRIGGER events this episode;
    once it reaches `y_max` the controller never arms again (SAFETY: at
    most Y_max yields/episode, never oscillate)."""

    def __init__(self, r_occ, d_stall, w, standoff, m, y_max):
        self.r_occ = r_occ
        self.d_stall = d_stall
        self.w = w
        self.standoff = standoff
        self.m = m
        self.y_max = y_max
        self.state = "idle"
        self.triggers = 0
        self.resume_streak = 0
        self.waypoints = []
        self.total_yields = 0
        self.total_steps = 0
        self.total_resumes = 0

    def armed(self):
        return self.triggers < self.y_max

    @staticmethod
    def is_stalled(replan_hist, d_stall):
        """STALLED(eef): net displacement < d_stall over a FULL window of W
        replan-boundary positions (replan_hist is maxlen-W; not yet full ==
        not enough history to judge -> not stalled)."""
        if len(replan_hist) < replan_hist.maxlen:
            return False
        return float(np.linalg.norm(
            np.asarray(replan_hist[-1], dtype=np.float64)
            - np.asarray(replan_hist[0], dtype=np.float64))) < d_stall

    @staticmethod
    def occupies(hazard_pos, target_pos, r_occ):
        """OCCUPIES(hazard, goal_region): hazard-to-corridor-target distance
        < r_occ. False (not occupied) if either position is unresolvable --
        the conservative default for both TRIGGER (won't arm) and RESUME
        (won't re-arm the occupied streak, i.e. treated as clear)."""
        if hazard_pos is None or target_pos is None:
            return False
        return float(np.linalg.norm(
            np.asarray(hazard_pos, dtype=np.float64)
            - np.asarray(target_pos, dtype=np.float64))) < r_occ

    @staticmethod
    def runtime_moving(hazard_hist, d_move):
        """RUNTIME MOVING(hazard) (fix round 2, G-Y2 forensics): the guard
        hazard's position has displaced more than `d_move` [m] across the
        last 2 replan-boundary readings of the SAME hazard identity.
        `hazard_hist` is a maxlen-2 deque of (replan_idx, hazard_pos)
        entries, reset by the caller whenever the tracked guard identity
        changes. Complements the settle-window `movers` set, which misses
        hazards that start moving AFTER the settle window (the LIBERO-
        Safety dynamic-intruder suite: the hand is stationary during the
        settle window and only starts moving once the episode is
        underway). Fewer than 2 readings -> not (yet) judged moving; a
        truly static hazard (t13-class) never displaces and stays
        excluded regardless of how many readings accumulate."""
        if len(hazard_hist) < 2:
            return False
        p_prev = np.asarray(hazard_hist[-2][1], dtype=np.float64)
        p_cur = np.asarray(hazard_hist[-1][1], dtype=np.float64)
        return float(np.linalg.norm(p_cur - p_prev)) > d_move

    def should_trigger(self, stalled, hazard_name, movers, occupies_flag,
                        runtime_moving_flag=False):
        """TRIGGER := stalled AND MOVING(hazard) AND OCCUPIES, only from
        "idle" and only while still armed (< y_max). MOVING(hazard) :=
        (hazard_name in the episode's settle-window `movers` set) OR
        `runtime_moving_flag` (fix round 2: a hazard that starts moving
        after the settle window also counts)."""
        moving = bool(movers is not None and hazard_name in movers) or bool(runtime_moving_flag)
        return bool(
            self.state == "idle" and self.armed() and stalled
            and hazard_name is not None and moving and occupies_flag)

    def trigger(self, eef_traj, hazard_pos):
        """Arm RETREAT: compute waypoints along the recorded trajectory back
        to standoff from the CURRENT hazard position. Degenerate case (no
        retreat needed / no trajectory to retreat along) goes straight to
        HOLD."""
        self.waypoints = _yield_retreat_waypoints(eef_traj, hazard_pos, self.standoff)
        self.triggers += 1
        self.total_yields += 1
        self.resume_streak = 0
        self.state = "retreat" if len(self.waypoints) >= 2 else "hold"
        return self.state

    def next_retreat_action(self, cur_pos, hazard_pos, hard_r,
                             exec_parity=False, raw_actions=False):
        """One certified RETREAT step. Returns (action | None, new_state).
        Certifies the segment [cur_pos, next_waypoint] against the CURRENT
        hazard_pos before executing; on certificate failure, does NOT
        execute the step -- switches to HOLD and returns (None, "hold")
        instead (never reuses the scripted +Z `in_retreat` lift)."""
        if self.state != "retreat" or len(self.waypoints) < 2:
            self.state = "hold"
            return None, self.state
        nxt = self.waypoints[1]
        if not _yield_certify_step(cur_pos, nxt, hazard_pos, hard_r):
            self.state = "hold"
            return None, self.state
        delta = np.asarray(nxt, dtype=np.float64) - np.asarray(cur_pos, dtype=np.float64)
        raw = np.concatenate([delta, np.zeros(4)])[np.newaxis, :]
        action = np.asarray(_plan_actions(
            raw, exec_parity=exec_parity, raw_actions=raw_actions))[0]
        self.waypoints.pop(0)
        self.total_steps += 1
        if len(self.waypoints) < 2:
            self.state = "hold"
        return action, self.state

    def hold_step(self):
        """Zero-action HOLD step."""
        self.total_steps += 1
        return np.zeros(7)

    def evaluate_resume(self, occupies_flag):
        """Call at each replan boundary while HOLDing. RESUME hysteresis: M
        consecutive ¬OCCUPIES evaluations clear Yield state back to idle; a
        single OCCUPIES resets the streak (no partial credit -- chatter
        prevention). Returns True iff this call resumed."""
        if occupies_flag:
            self.resume_streak = 0
            return False
        self.resume_streak += 1
        if self.resume_streak >= self.m:
            self.state = "idle"
            self.waypoints = []
            self.resume_streak = 0
            self.total_resumes += 1
            return True
        return False


def _stall_signal(send_stall_signal, stall_hist, d_stall, step_guard):
    """Task 5 flag A client-side `stalled` bit for the guidance payload.
    Pure given the history deque (does not mutate it -- the caller appends
    the current eef position before calling, same as the Yield-regime call
    site above). Reuses `YieldController.is_stalled` -- the EXISTING
    detection-only static method (net eef displacement across the replan-
    boundary window < d_stall) -- WITHOUT any of stall_recovery's/Yield's
    own response machinery (HDC detour, scripted lift, certified retreat).
    False whenever the flag is off, no guard is currently engaged
    (`step_guard` empty), or the window hasn't filled yet."""
    if not send_stall_signal:
        return False
    return bool(YieldController.is_stalled(stall_hist, d_stall) and step_guard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="obstacle_avoidance")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--num_trials_per_task", type=int, default=5)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8150)
    ap.add_argument("--disable_guidance", action="store_true")
    ap.add_argument("--obstacle_id_source", choices=["gt", "symbolic", "fol", "folprop", "folpropvlm"], default="gt")
    ap.add_argument("--fol_ablate", choices=["none", "geom_only", "no_exempt", "no_class", "no_heat_gate"],
                    default="none",
                    help="Ablation variants of the FOL hazard election "
                         "(symbolic_identity.ablated_obstacle_id), for the "
                         "paper's ablation study. Each removes ONE conjunct "
                         "of HAZARD(x) := PROTECTED(x) | MOVING(x) | "
                         "(PRED(x) & ~MENTIONED(x) & ~TARGET(x)). Only "
                         "applies when --obstacle_id_source is fol/folprop/"
                         "folpropvlm; documented no-op for gt/symbolic. "
                         "Default 'none' = byte-identical to the incumbent "
                         "identity function.")
    ap.add_argument("--entity_pos_source", choices=["gt", "percep"], default="gt",
                    help="'percep' = episode-start RGB-D back-projection "
                         "(GroundingDINO regions) for ALL positions consumed by "
                         "ident + guidance. Static snapshot: dynamic hazards "
                         "keep their settle-end estimate (documented no-GT tax)")
    ap.add_argument("--percep_z_correction", type=float, default=-0.03)
    ap.add_argument("--percep_render_res", type=int, default=256,
                    help="Render resolution for percep localization (depth/"
                         "RGB fed to the detector). 512 halves pixel-"
                         "quantization error at one extra render per object "
                         "per episode (arm G).")
    ap.add_argument("--percep_bbox_shrink", type=float, default=0.0,
                    help="Shrink detector bboxes toward center by this "
                         "fraction before depth sampling (arm G: cuts the "
                         "systematic background-depth centroid bias; 0 = "
                         "full rectangle).")
    ap.add_argument("--percep_refresh_alpha", type=float, default=0.0,
                    help="With --percep_refresh_movers: EMA weight for the "
                         "mover track (0 = raw replacement, the arm-F mode "
                         "that LOST to the static snapshot via detector "
                         "noise). alpha in (0,1] blends each accepted "
                         "re-detection into a smooth track.")
    ap.add_argument("--percep_refresh_gate", type=float, default=0.15,
                    help="Innovation gate (meters) for filtered refresh: a "
                         "re-detection farther than this from the current "
                         "track is treated as a detector outlier and "
                         "discarded (track kept).")
    ap.add_argument("--percep_refresh_cameras", type=str, default="",
                    help="Cameras for --percep_refresh_movers re-perception "
                         "(comma-separated; empty = same as "
                         "--entity_cameras). Arm F3: robot0_eye_in_hand "
                         "first — at close range (where the freezes happen) "
                         "the wrist view sees the hand large and unoccluded, "
                         "exactly where the agentview detector is weakest; "
                         "with --entity_view_tau the FIRST camera wins on "
                         "cross-view disagreement.")
    ap.add_argument("--percep_refresh_movers", action="store_true",
                    help="No-GT tier: re-run percep localization for MOVER "
                         "entities at every chunk boundary, so dynamic "
                         "hazards (hands) are guarded at their CURRENT "
                         "position instead of the settle-end snapshot (the "
                         "documented no-GT tax on dynamic hazards). Static "
                         "entities keep the snapshot; guard membership and "
                         "identity are unchanged. Default off = "
                         "byte-identical.")
    ap.add_argument("--entity_view_tau", type=float, default=0.0,
                    help="Consistency-gated multi-view fusion (meters; 0 = "
                         "off = blind mean). Views are mean-fused only when "
                         "their estimates agree within tau; on disagreement "
                         "the first --entity_cameras view wins alone (arm B "
                         "forensics: blind birdview mean collapsed t0/t10)")
    ap.add_argument("--entity_cameras", type=str, default="agentview",
                    help="Comma-separated cameras fused for percep entity "
                         "localization (SafeLIBERO probe 42844384: +birdview "
                         "halves xy error; cameras absent from the model are "
                         "skipped per-view)")
    ap.add_argument("--percep_mover_threshold", type=float, default=0.06,
                    help="settle-window displacement [m] marking MOVING(x) from "
                         "two percep snapshots (looser than the GT 0.01 — "
                         "correlated detector noise, med abs err 0.067)")
    ap.add_argument("--corridor", action="store_true",
                    help="Accepted for CLI parity with the SafeLIBERO runner. "
                         "NOTE: this client already sends the corridor anchors "
                         "(target_pos/dest_pos, re-read each replan) by "
                         "default since the HRI forensics fix, so passing the "
                         "flag does not change the payload.")
    ap.add_argument("--guard_topk", type=int, default=1,
                    help="Top-k guard set for the symbolic/fol/folprop/"
                         "folpropvlm identity "
                         "sources (guard_set.py runner-up sent as obstacle2; "
                         "the server accepts at most one secondary guard). "
                         "1 = historical single-guard behavior. The gt source "
                         "always guards every CheckRobotContact hazard.")
    ap.add_argument("--dynamic_election", action="store_true",
                    help="Phase 2 guard de-election (client-only, oa): "
                         "per replan, re-evaluate whether each episode-"
                         "elected guard entry still satisfies its electing "
                         "FOL disjunct. PROTECTED/MOVING entries are always "
                         "retained; a NEAR_PATH-only entry (fol theory: "
                         "HAZARD(x) := PROTECTED(x) | MOVING(x) | "
                         "(~MENTIONED(x) & NEAR_PATH(x))) is released once "
                         "the eef->target segment no longer passes near it, "
                         "re-entering automatically if the segment swings "
                         "back (stateless filter, not a latch). Composes "
                         "BEFORE --engage_radius/--engage_cone/"
                         "--holding_disengage (applied to their input). "
                         "RELEASE-ONLY relative to the episode election set "
                         "-- never adds a hazard election did not pick. "
                         "No-op unless --obstacle_id_source=fol (the only "
                         "theory with a NEAR_PATH disjunct). Default off = "
                         "byte-identical.")
    ap.add_argument("--engage_radius", type=float, default=0.0,
                    help="Minimum-intervention engagement gate [m]. At each "
                         "server query a guard is included in the guidance "
                         "payload only if dist(eef, guard) <= R (primary and "
                         "obstacle2 gated independently); if no guard is "
                         "within R the query carries no guidance block — "
                         "identical to the --disable_guidance / no-obstacle "
                         "path. Corridor anchors ride the guidance block, so "
                         "they are sent whenever any guard is engaged. "
                         "pi0.5 executes replan_steps-action chunks, so "
                         "gating decisions happen at chunk boundaries (each "
                         "query), not per env step. 0 = disabled, legacy "
                         "always-on guarding (byte-identical payloads). "
                         "Composes with --duality_disengage (episode-level "
                         "guard clearing runs first) and --guard_topk (each "
                         "ranked guard gated on its own live position).")
    ap.add_argument("--engage_cone", type=float, default=None,
                    help="APPROACHING(eef, guard) velocity gate [deg]. "
                         "Composes AFTER --engage_radius (applied to its "
                         "output): a guard entry is kept iff dist(eef, "
                         "guard) <= ENGAGE_CONE_HARD_R (0.14 m, always "
                         "engaged at close range) OR the eef velocity "
                         "(finite difference between consecutive "
                         "chunk-boundary queries) is closing on it within "
                         "this many degrees. First query (no velocity yet) "
                         "is the conservative fallback: every entry is "
                         "engaged. None (default) = disabled, legacy "
                         "always-on guarding (byte-identical payloads). "
                         "Corridor anchors are never gated.")
    ap.add_argument("--holding_disengage", action="store_true",
                    help="HOLDING(eef) gripper-state suppression. Composes "
                         "AFTER --engage_radius/--engage_cone (applied to "
                         "their output): while the gripper's finger-opening "
                         "width (robot0_gripper_qpos) has been in the "
                         f"closed-on-something band ({HOLDING_WIDTH_MIN}, "
                         f"{HOLDING_WIDTH_MAX}) m for >= 2 consecutive "
                         "chunk-boundary queries, STATIC guard entries "
                         "(name not in the episode's movers set) are "
                         "dropped; mover/hand entries always stay — this "
                         "flag can never suppress a dynamic hazard. If the "
                         "movers metadata is unavailable the fallback is "
                         "conservative: suppress nothing. Default off = "
                         "byte-identical legacy behavior.")
    ap.add_argument("--mover_lead", type=float, default=None,
                    help="MOVER_LEAD(hazard) predictive barrier anchor "
                         "[env steps]. Composes AFTER --engage_radius/"
                         "--engage_cone/--holding_disengage (applied to "
                         "whichever guard entries survive gating -- "
                         "adjusts positions, not membership): for entries "
                         "classified as movers (episode movers metadata, "
                         "as in --holding_disengage), replaces the "
                         "position sent to the server with the "
                         "constant-velocity extrapolation pos + v*STEPS, "
                         "where v is the per-entity per-step velocity from "
                         "finite differences between consecutive "
                         "chunk-boundary queries (caller tracks previous "
                         "positions per entity name and elapsed env steps "
                         "between queries to normalize v). Extrapolation "
                         f"displacement capped at {MOVER_LEAD_CAP} m "
                         "(MOVER_LEAD_CAP). First query for an entity (no "
                         "previous sample yet) -> observed position "
                         "unchanged. Static entries unchanged. Adjustment "
                         "is used ONLY for the guard payload sent to the "
                         "server -- logging/violation accounting keep "
                         "using observed positions; corridor anchors "
                         "(target_pos/dest_pos) are also EXEMPT even if a "
                         "guard entry's name equals the destination (HRI "
                         "duality). With --ssm_margins, the speed-and-"
                         "separation margin is computed from the "
                         "extrapolated (not observed) primary guard "
                         "position, since the barrier is now anchored "
                         "predictively. None (default) = disabled, "
                         "byte-identical legacy payloads.")
    ap.add_argument("--safety_prompt", action="store_true",
                    help="Append an avoidance clause naming the identified hazard to the prompt (pi0.7-style inference-time instruction)")
    ap.add_argument("--refuse_unsafe", action="store_true",
                    help="E4 judge refuses semantically unsafe instructions")
    ap.add_argument("--results_output_dir", default="ls_results")
    ap.add_argument("--save_videos", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="Override the per-level step cap (diagnostics)")
    ap.add_argument("--duality_disengage", action="store_true",
                    help="Hazard-destination duality routing: if the guarded "
                         "hazard IS (or overlaps, <0.15 m) the parsed "
                         "destination, disable geometric keep-out for the "
                         "episode — HRI forensics: baseline violations are 0 "
                         "and keep-out only destroys success (54->18 TSR)")
    ap.add_argument("--exec_parity", action="store_true",
                    help="Execution parity with the authors' LIBERO-Safety "
                         "harness (run_libsafety_eval_openpi.py): (1) patch "
                         "robosuite's OSC_POSE gains to the vendored "
                         "LIBERO-Safety values (output_max 2, kp 750 — vs "
                         "stock 0.05/150; the ~2.2-2.5x env-step slowdown "
                         "root cause), (2) raw action passthrough (no "
                         "ACTION_TO_CMD, no client clip), (3) client-side "
                         "resize_with_pad to 224 + uint8 exactly as the "
                         "baseline client. Default off = byte-identical "
                         "legacy behavior. Composes with --duality_disengage/"
                         "--guard_topk/--engage_radius (guidance payloads are "
                         "metric positions, untouched by execution parity); "
                         "supersedes --raw_actions when both are set. "
                         "Guidance-arm note: keep the server at "
                         "--translation_scale 1.0 --cmd_clip 0.05 for the "
                         "no-guidance arm; for guided arms consider "
                         "--translation_scale 2.0 --cmd_clip 0.025 so the "
                         "barrier's actuation model tracks the vendored "
                         "controller's ~2x realized motion per unit action.")
    ap.add_argument("--raw_actions", action="store_true",
                    help="Checkpoint outputs [-1,1] commands (off-the-shelf "
                         "pi05_libero) — skip the metric ACTION_TO_CMD "
                         "conversion that the LS finetune needs")
    ap.add_argument("--replan_steps", type=int, default=5,
                    help="Actions executed per chunk before replanning (10 = "
                         "openpi's default full-chunk LIBERO consumption)")
    ap.add_argument("--camera_res", type=int, default=256,
                    help="Env render resolution (gap diagnosis: their data "
                         "pipeline is 128px)")
    # Runtime steering revision (spec 2026-07-27) — client-side controllers.
    ap.add_argument("--dual_eta", action="store_true",
                    help="B: dual-adaptive repulsor strength from the server's "
                         "selected_margin diag (pair with --repair_schedule none server)")
    ap.add_argument("--dual_kappa", type=float, default=0.5)
    ap.add_argument("--dual_mref", type=float, default=0.02)
    ap.add_argument("--dual_eta_max", type=float, default=0.01)
    ap.add_argument("--stall_recovery", action="store_true",
                    help="A2: stall -> HDC detour replans -> scripted lift-retreat")
    ap.add_argument("--send_stall_signal", action="store_true",
                    help="Task 5 flag A (server-side --stall_resample): add a "
                         "boolean `stalled` field to the guidance payload, "
                         "gated on a guard being engaged AND the reused "
                         "YieldController.is_stalled detector (net eef "
                         "displacement below --yield_d_stall over the last "
                         "--yield_w replan-boundary positions) firing -- "
                         "detection only, independent of --yield_regime/"
                         "--stall_recovery's own responses. Default off: "
                         "payload is byte-identical to today.")
    ap.add_argument("--yield_regime", action="store_true",
                    help="Phase 1 Yield (client-only, certified retreat): "
                         "TRIGGER (STALLED(eef) AND MOVING(hazard) AND "
                         "OCCUPIES(hazard, goal) within --yield_r_occ) -> "
                         "RETREAT along the episode's own executed eef "
                         "trajectory to --yield_standoff, each step "
                         "re-certified against the CURRENT hazard position "
                         "(ENGAGE_CONE_HARD_R) before executing -> HOLD "
                         "(zero-action, re-evaluating OCCUPIES every replan) "
                         "-> RESUME after --yield_m consecutive replans of "
                         "¬OCCUPIES. At most --yield_max triggers per "
                         "episode. Client-only: the guided server is "
                         "stateless and untouched. Does NOT reuse the "
                         "scripted +Z `in_retreat` lift; mutually exclusive "
                         "with --stall_recovery. Default off = "
                         "byte-identical.")
    ap.add_argument("--yield_r_occ", type=float, default=0.20,
                    help="OCCUPIES(hazard, goal_region) radius [m]: hazard-"
                         "to-corridor-target distance. Its own declared "
                         "constant -- distinct from ENGAGE_CONE_HARD_R "
                         "(0.14, eef-to-hazard, used for retreat "
                         "certification). Raised 0.12 -> 0.20 (fix round 2, "
                         "G-Y2 forensics): the hand sits within/near a "
                         "0.14 m hard core of the goal region and "
                         "corridor_target is offset from where it hovers, "
                         "so 0.12 under-covered OCCUPIES even in single-"
                         "mover episodes.")
    ap.add_argument("--yield_d_stall", type=float, default=0.01,
                    help="STALLED(eef): net eef displacement [m] threshold "
                         "over the last --yield_w replan-boundary "
                         "positions.")
    ap.add_argument("--yield_d_move", type=float, default=0.01,
                    help="RUNTIME MOVING(hazard) (fix round 2, G-Y2 "
                         "forensics): displacement [m] threshold across the "
                         "last 2 replan-boundary readings of the tracked "
                         "guard hazard's position. Complements the "
                         "settle-window `movers` set -- covers hazards "
                         "(e.g. the dynamic-intruder hand) that start "
                         "moving AFTER the settle window, which `movers` "
                         "alone misses. A truly static hazard never "
                         "displaces and stays excluded.")
    ap.add_argument("--yield_w", type=int, default=5,
                    help="STALLED(eef) window, in REPLANS (chunk-boundary "
                         "queries) -- distinct from --stall_recovery's "
                         "40-env-step window.")
    ap.add_argument("--yield_standoff", type=float, default=0.20,
                    help="RETREAT target distance [m] from the CURRENT "
                         "hazard position.")
    ap.add_argument("--yield_m", type=int, default=2,
                    help="RESUME hysteresis: consecutive replan-boundary "
                         "¬OCCUPIES evaluations required to exit Yield "
                         "and return to normal guided policy.")
    ap.add_argument("--yield_max", type=int, default=2,
                    help="Y_max: Yield may TRIGGER at most this many times "
                         "per episode; after the cap it never re-arms "
                         "(reverts to current freeze behavior).")
    ap.add_argument("--ssm_margins", action="store_true",
                    help="A3: speed-and-separation margin for DYNAMIC hazards (ISO/TS "
                         "15066 shape): r = base - 0.03 + ssm_k * closing_speed, "
                         "clamped [base-0.03, base+0.05]. Static hazards keep the "
                         "legacy fixed radius. Targets the human_safety stall tax.")
    ap.add_argument("--ssm_k", type=float, default=0.5,
                    help="Margin gain [m per m/s of hazard closing speed].")
    ap.add_argument("--adaptive_replan", action="store_true",
                    help="Replan every 2 control steps while certified clearance "
                         "< --adaptive_clearance (reactivity vs the moving hand).")
    ap.add_argument("--adaptive_clearance", type=float, default=0.12)
    ap.add_argument("--log_trajectories", action="store_true",
                    help="Save executed actions of SAFE SUCCESSFUL episodes (npz) "
                         "for self-distillation finetuning.")
    ap.add_argument("--task_filter", type=int, nargs="*", default=None,
                    help="Restrict to these task_ids (diagnostics)")
    ap.add_argument("--log_transitions", action="store_true",
                    help="Append one JSON line per env.step() to "
                         "<results_output_dir>/transitions_<suite>_L<level>.jsonl: "
                         "task/ep/t, eef, grip, the action sent to env, and "
                         "entity positions (the runner's existing GT/percep "
                         "position source — no new perception). File opened "
                         "lazily on first write, flushed per episode. Default "
                         "off = zero overhead (no record built, no file).")
    ap.add_argument("--log_percep_frames", default=None,
                    help="Directory to save one compressed record per replan "
                         "boundary (RGB-D + camera K/T + ALL GT entity "
                         "positions + eef_pos/t/guard) as training data for "
                         "the Phase-3 learned hazard localizer -- labels are "
                         "free (GT positions). Default None = zero overhead: "
                         "no writer object created, no behavioral or output "
                         "change on the control path (pure observation read "
                         "+ file write when on).")
    args = ap.parse_args()

    assert not (args.yield_regime and args.stall_recovery), (
        "--yield_regime and --stall_recovery are mutually exclusive: "
        "Yield's certified retreat replaces --stall_recovery's scripted "
        "+Z lift-retreat, it does not compose with it.")

    if args.exec_parity:
        # Must precede OffScreenRenderEnv creation: the wrapper snapshots the
        # controller config at env construction time.
        _install_parity_controller()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    gdino_detector = None
    if args.entity_pos_source == "percep":
        from percep_obstacle import make_gdino_detector, estimate_object_positions
        gdino_detector = make_gdino_detector()
        logging.info("GroundingDINO detector service started (LS Tier-Percep)")

        def percep_snapshot(env, names):
            return estimate_object_positions(
                env.sim, names, cameras=tuple(args.entity_cameras.split(",")),
                camera_height=args.percep_render_res,
                camera_width=args.percep_render_res,
                region_source="detector", detector=gdino_detector,
                z_correction=args.percep_z_correction,
                consistency_tau=(args.entity_view_tau or None),
                bbox_shrink=args.percep_bbox_shrink)
    bm = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [i for i in range(bm.get_num_tasks())
                if getattr(bm.get_task(i), "level", None) == args.level]
    if args.task_filter is not None:
        task_ids = [i for i in task_ids if i in args.task_filter]
    logging.info(f"{args.suite} L{args.level}: {len(task_ids)} tasks")

    out_dir = pathlib.Path(args.results_output_dir) / args.suite / f"L{args.level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    episodes_log = out_dir / f"episodes_{stamp}.jsonl"
    # --log_transitions: file handle opened lazily on first write (main()-scope
    # so it stays open across tasks/episodes; one file for the whole run).
    transitions_fh = None
    transitions_path = (pathlib.Path(args.results_output_dir)
                         / f"transitions_{args.suite}_L{args.level}.jsonl")

    # --log_percep_frames: default-off hazard-localizer training-data logger.
    # No writer/manifest object exists at all when the flag is absent (pure
    # zero-overhead default-off, mirrors --log_transitions above).
    percep_frame_manifest_skipped = None
    percep_frame_manifest_path = None
    percep_frame_error_stats = None
    if args.log_percep_frames:
        percep_frame_manifest_skipped = set()
        percep_frame_error_stats = {"write_errors": 0}
        percep_frame_dir = (pathlib.Path(args.log_percep_frames)
                             / args.suite / f"L{args.level}")
        percep_frame_dir.mkdir(parents=True, exist_ok=True)
        percep_frame_manifest_path = percep_frame_dir / "manifest.json"

    if args.refuse_unsafe:
        from e4_ssr_offline import is_unsafe

    per_task = []
    _warned_fol_ablate_noop = [False]  # log the --fol_ablate no-op once, not per episode
    for task_id in task_ids:
        task = bm.get_task(task_id)
        desc = str(task.language)
        # Level-aware paths (smoke finding 2026-07-20: the flat
        # problem_folder/bddl_file path misses the level directory).
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=bm.get_task_bddl_file_path_by_level_id(task.level, task.level_id),
                camera_heights=args.camera_res, camera_widths=args.camera_res,
            )
        except (FileNotFoundError, AssertionError) as e:
            # Some reasoning_safety L2 bddls are absent from the release
            # (crashed job 42587320) — skip the task, keep the run alive.
            logging.warning(f"SKIP task {task_id}: missing bddl ({e})")
            continue
        try:
            init_states = bm.get_task_init_states_by_level_id(task.level, task.level_id)
        except FileNotFoundError:
            init_states = None  # reasoning_safety ships no .pruned_init
        logging.info(f"=== task {task_id} (L{args.level}): {desc}")

        refused = args.refuse_unsafe and is_unsafe(desc)
        succ = viol = 0
        for ep in range(args.num_trials_per_task):
            if refused:
                with open(episodes_log, "a") as ef:
                    ef.write(json.dumps({"task": task_id, "ep": ep, "refused": True,
                                         "success": False, "violation": False}) + "\n")
                continue
            obs = env.reset()
            if init_states is not None:
                obs = env.set_init_state(init_states[ep % len(init_states)])
            pos_settle_start = None
            percep_a = None
            for wi in range(NUM_STEPS_WAIT):
                obs, _, _, _ = env.step(DUMMY_ACTION)
                if wi == 4:  # after physics settling, before the window ends
                    pos_settle_start = object_positions(obs)
                    if args.entity_pos_source == "percep":
                        percep_a = percep_snapshot(env, list(pos_settle_start))
            percep_pos = None
            if args.entity_pos_source == "percep":
                percep_pos = percep_snapshot(env, list(object_positions(obs)))
                movers = {n for n, p in (percep_a or {}).items()
                          if n in percep_pos and np.linalg.norm(
                              np.asarray(percep_pos[n]) - np.asarray(p))
                          > args.percep_mover_threshold}
                logging.info(f"  [entity-percep] ep {ep} localized "
                             f"{len(percep_pos)}/{len(object_positions(obs))}")
            else:
                movers = detect_movers(pos_settle_start or {}, object_positions(obs))
            if movers:
                logging.info(f"  [moving] ep {ep}: {sorted(movers)}")

            hazards = hazard_names_from_constraints(env)
            # Identity candidates: percep snapshot when no-GT, live obs else.
            _pos_src = percep_pos if percep_pos is not None else object_positions(obs)
            cands = {k: np.asarray(v) for k, v in _pos_src.items()
                     if WORKSPACE[0][0] < v[0] < WORKSPACE[0][1]
                     and WORKSPACE[1][0] < v[1] < WORKSPACE[1][1]
                     and v[2] > 0}
            eef = np.asarray(obs["robot0_eef_pos"])  # proprio: allowed in no-GT
            if (args.fol_ablate != "none"
                    and args.obstacle_id_source not in ("fol", "folprop", "folpropvlm")
                    and not _warned_fol_ablate_noop[0]):
                logging.warning(
                    f"--fol_ablate={args.fol_ablate} is a no-op for "
                    f"--obstacle_id_source={args.obstacle_id_source} "
                    "(only fol/folprop/folpropvlm are affected)")
                _warned_fol_ablate_noop[0] = True
            guard = _topk_guard(
                args.obstacle_id_source, desc, cands, eef, movers,
                [h for h in hazards if f"{h}_pos" in obs], k=args.guard_topk,
                fol_ablate=args.fol_ablate)
            ident_correct = (bool(guard) and guard[0] in hazards) if hazards else None
            # pi0.7-style inference-time safety instruction: append an
            # avoidance clause naming the identified hazard to the PROMPT
            # itself — steering via the policy's own language conditioning.
            prompt_text = desc
            if args.safety_prompt and guard:
                haz = re.sub(r"(__\d+)?(_\d+)?$", "", guard[0]).replace("_", " ")
                prompt_text = f"{desc}, while staying away from the {haz}"
                if ep == 0:
                    logging.info(f"  [safety-prompt] '{prompt_text}'")
            # Corridor exemption anchors: sanctioned approach to target and
            # destination (destination may BE the guarded hazard — HRI).
            # Always on in this client (see --corridor help).
            corridor_target, corridor_dest = _corridor_anchor(desc, cands)
            if (args.duality_disengage and guard and corridor_dest is not None
                    and corridor_dest in cands and guard[0] in cands):
                _ddist = float(np.linalg.norm(np.asarray(cands[guard[0]])
                                              - np.asarray(cands[corridor_dest])))
                if guard[0] == corridor_dest or _ddist < 0.15:
                    logging.info(f"  [duality] ep {ep}: hazard {guard[0]} ~ "
                                 f"dest {corridor_dest} (d={_ddist:.3f}) -> "
                                 "keep-out disengaged")
                    guard = []

            plan = collections.deque()
            done = False
            violated = False
            t = 0
            percep_frame_replan_idx = 0
            frames = []
            eef_path_len = 0.0
            prev_eef = np.asarray(obs["robot0_eef_pos"]).copy()
            cone_prev_eef = None  # --engage_cone: velocity grounded across chunk-boundary queries only
            holding_streak = 0  # --holding_disengage: consecutive band queries
            lead_prev_pos = {}  # --mover_lead: last chunk-boundary observed pos per mover name
            lead_steps_since_query = 0  # --mover_lead: env steps elapsed since last chunk-boundary query
            # Revision controllers (spec 2026-07-27): per-episode state.
            dual_lam = 0.0
            ep_actions = []                           # self-distillation log
            eef_hist = collections.deque(maxlen=40)   # A2 stall window
            # Task 5 flag A (--send_stall_signal): independent replan-
            # boundary window, reusing YieldController.is_stalled's window
            # size (--yield_w) regardless of --yield_regime.
            ab_stall_hist = (collections.deque(maxlen=args.yield_w)
                              if args.send_stall_signal else None)
            stall_hdc_replans = 0
            in_retreat = 0
            ep_stalls = ep_retreats = 0
            ssm_prev_g0 = None                        # A3 hazard tracking
            ssm_init_g0 = None
            ssm_moved = False
            # Phase 1 Yield regime (client-only, certified retreat): per-
            # episode state. `eef_traj` records EVERY env step (never
            # cleared mid-episode, unlike `eef_hist`) so RETREAT can walk
            # the episode's own executed path; `yield_replan_hist` records
            # only replan-boundary positions for STALLED(eef).
            yc = (YieldController(args.yield_r_occ, args.yield_d_stall,
                                   args.yield_w, args.yield_standoff,
                                   args.yield_m, args.yield_max)
                  if args.yield_regime else None)
            eef_traj = collections.deque(maxlen=600) if args.yield_regime else None
            yield_replan_hist = (collections.deque(maxlen=args.yield_w)
                                  if args.yield_regime else None)
            yield_hazard_name = None  # hazard identity captured at TRIGGER
            # RUNTIME MOVING (fix round 2): (replan_idx, pos) history of the
            # CURRENT candidate guard hazard, reset on identity change.
            yield_hazard_track_name = None
            yield_hazard_hist = collections.deque(maxlen=2) if args.yield_regime else None
            # Blocker telemetry (fix round 2, diagnosability): counts, over
            # replan-boundary evaluations with a guard engaged, how many
            # times each TRIGGER conjunct was the SOLE blocker.
            yield_blockers = ({"stall": 0, "moving": 0, "occupies": 0}
                               if args.yield_regime else None)
            max_steps = args.max_steps or MAX_STEPS[args.level]
            while t < max_steps and not done:
                _yield_active = _yield_intercepts(yc)
                if not _yield_active and not plan:
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    if args.save_videos:
                        frames.append(img)
                    if args.exec_parity:
                        img, wrist = _parity_images(img, wrist)
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist,
                        "observation/state": np.concatenate((
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )),
                        "prompt": prompt_text,
                    }
                    # Fix round 3 (G-Y2 forensics): the refresh set is
                    # movers-only when Yield is off (byte-identical legacy
                    # behavior, including the no-op-when-empty case); with
                    # --yield_regime on, the current guard hazard names are
                    # ALSO refreshed, since the settle-window `movers` set
                    # misses hazards that start moving after the settle
                    # window (see `_percep_refresh_set`).
                    _refresh = _percep_refresh_set(movers, guard, args.yield_regime)
                    if (args.percep_refresh_movers and percep_pos is not None
                            and _refresh):
                        # Per-replan mover re-perception (2026-08-08): refresh
                        # ONLY mover entities so the dynamic-hazard barrier
                        # anchors to the CURRENT hand position. A failed
                        # re-localization keeps the previous estimate
                        # (fail-safe: never un-guards on a perception miss).
                        # Arm-F forensics: RAW replacement (alpha=0) is
                        # noisier than the static snapshot and REGRESSED
                        # L1 44.4->34.0; alpha>0 runs a gated EMA track
                        # instead -- outlier detections (innovation beyond
                        # --percep_refresh_gate) are discarded, accepted ones
                        # blend smoothly.
                        # Arm F3: --percep_refresh_cameras overrides the view
                        # set for the refresh only (wrist-primary tracking —
                        # close range is where agentview fails and freezes
                        # happen); empty = entity_cameras via percep_snapshot.
                        if args.percep_refresh_cameras:
                            _fresh = estimate_object_positions(
                                env.sim, sorted(_refresh),
                                cameras=tuple(args.percep_refresh_cameras.split(",")),
                                region_source="detector", detector=gdino_detector,
                                z_correction=args.percep_z_correction,
                                consistency_tau=(args.entity_view_tau or None))
                        else:
                            _fresh = percep_snapshot(env, sorted(_refresh))
                        _a = args.percep_refresh_alpha
                        if _a <= 0:
                            percep_pos.update(_fresh)
                        else:
                            for _n, _p in _fresh.items():
                                _prev = percep_pos.get(_n)
                                _p = np.asarray(_p, dtype=np.float64)
                                if _prev is None:
                                    percep_pos[_n] = _p
                                    continue
                                _prev = np.asarray(_prev, dtype=np.float64)
                                if np.linalg.norm(_p - _prev) > args.percep_refresh_gate:
                                    continue  # detector outlier: keep track
                                percep_pos[_n] = (1.0 - _a) * _prev + _a * _p

                    def _guard_pos(name):
                        # GT tier: LIVE obs every replan (tracks :dynamics
                        # movers). Percep tier: settle-end snapshot, movers
                        # optionally refreshed per replan above
                        # (--percep_refresh_movers).
                        if percep_pos is not None:
                            return percep_pos.get(name)
                        return obs.get(f"{name}_pos")

                    # Dynamic guard de-election (Phase 2, oa): per replan,
                    # re-evaluate NEAR_PATH-only episode-elected entries
                    # against the CURRENT eef->target segment; runs BEFORE
                    # the engagement gates below so `step_guard` seen by
                    # cone/holding/yield logic is already the filtered list.
                    # Default off leaves `guard` untouched (byte-identical).
                    _elected = guard
                    if args.dynamic_election:
                        _elected = _dynamic_deelect(
                            guard, args.obstacle_id_source, _guard_pos,
                            obs["robot0_eef_pos"],
                            _guard_pos(corridor_target) if corridor_target else None,
                            movers)

                    # Engagement gate: per-query (chunk boundary) distance
                    # gating of each guard entry; R=0 returns guard unchanged.
                    step_guard = _gate_guard(
                        _elected, _guard_pos, obs["robot0_eef_pos"],
                        args.engage_radius)
                    if args.engage_cone is not None:
                        step_guard = _cone_gate(
                            step_guard, _guard_pos, obs["robot0_eef_pos"],
                            cone_prev_eef, args.engage_cone, ENGAGE_CONE_HARD_R)
                    cone_prev_eef = np.asarray(
                        obs["robot0_eef_pos"], dtype=np.float64).copy()
                    if args.holding_disengage:
                        _gw = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
                        _width = float(abs(_gw[0]) + abs(_gw[1]))
                        holding_streak = (
                            holding_streak + 1
                            if HOLDING_WIDTH_MIN < _width < HOLDING_WIDTH_MAX
                            else 0)
                        step_guard = _holding_filter(
                            step_guard, movers, _is_holding(_width, holding_streak))
                    # Task 5 flag A (--send_stall_signal): append THEN
                    # evaluate, same order as the Yield-regime call site
                    # below, over `step_guard` as finally gated above.
                    if args.send_stall_signal:
                        ab_stall_hist.append(
                            np.asarray(obs["robot0_eef_pos"], dtype=np.float64))
                    _ab_stalled = _stall_signal(
                        args.send_stall_signal, ab_stall_hist,
                        args.yield_d_stall, step_guard)
                    if args.yield_regime and yc.state == "idle":
                        # TRIGGER evaluation, replan boundaries only.
                        # STALLED(eef): net displacement < d_stall over the
                        # last --yield_w replan-boundary eef positions, AND
                        # a guard is currently engaged (step_guard non-empty
                        # after the gating chain above).
                        yield_replan_hist.append(
                            np.asarray(obs["robot0_eef_pos"], dtype=np.float64))
                        _y_stalled = (YieldController.is_stalled(
                            yield_replan_hist, args.yield_d_stall)
                            and bool(step_guard))
                        _y_candidate = step_guard[0] if step_guard else None
                        _y_hpos = _guard_pos(_y_candidate) if _y_candidate else None
                        # RUNTIME MOVING (fix round 2): track the CURRENT
                        # candidate hazard's position across replan
                        # boundaries; reset the history whenever the
                        # tracked identity changes.
                        if _y_candidate != yield_hazard_track_name:
                            yield_hazard_track_name = _y_candidate
                            yield_hazard_hist.clear()
                        if _y_candidate is not None and _y_hpos is not None:
                            yield_hazard_hist.append((t, _y_hpos))
                        _y_runtime_moving = YieldController.runtime_moving(
                            yield_hazard_hist, args.yield_d_move)
                        _y_ctgt, _ = _corridor_anchor(desc, cands)
                        _y_ctgt_pos = _guard_pos(_y_ctgt) if _y_ctgt else None
                        _y_occ = YieldController.occupies(
                            _y_hpos, _y_ctgt_pos, args.yield_r_occ)
                        if step_guard:
                            # Blocker telemetry (diagnostic-only, does not
                            # affect control flow): which single conjunct
                            # (if exactly one) blocked TRIGGER this
                            # evaluation, via the SAME pure function tested
                            # directly by the CPU test suite.
                            _y_moving_flag = bool(
                                movers is not None and _y_candidate in movers
                            ) or _y_runtime_moving
                            _y_blocker = _yield_blocker_conjunct(
                                _y_stalled, _y_moving_flag, _y_occ)
                            if _y_blocker is not None:
                                yield_blockers[_y_blocker] += 1
                        if yc.should_trigger(_y_stalled, _y_candidate, movers,
                                              _y_occ, _y_runtime_moving):
                            yield_hazard_name = _y_candidate
                            yc.trigger(list(eef_traj), _y_hpos)
                            # Stall window is frozen during RETREAT/HOLD (not
                            # appended to while yc.state != "idle") and must
                            # not carry stale pre-yield positions into the
                            # next idle-phase stall evaluation -- clear it.
                            yield_replan_hist.clear()
                            logging.info(
                                f"  [yield] t={t}: TRIGGER hazard="
                                f"{yield_hazard_name} -> {yc.state} "
                                f"({yc.triggers}/{args.yield_max})")
                    # MOVER_LEAD(hazard): predictive barrier anchor, applied
                    # AFTER gating to whichever entries survived it (adjusts
                    # positions, not membership). `_lead_pos`/`_lead_anchor_pos`
                    # default to the unadjusted `_guard_pos` lookup so the
                    # payload path is byte-identical when the flag is off.
                    # `_lead_anchor_pos` additionally exempts the corridor
                    # target/dest names (Task 1/2 contract: corridor anchors
                    # are NEVER gated/adjusted here, even if a guard entry's
                    # name happens to equal the destination — HRI duality,
                    # e.g. "the hand is hazard AND destination").
                    _lead_pos = _lead_anchor_pos = _guard_pos
                    if args.mover_lead is not None:
                        _cur_lead_map = {n: _guard_pos(n) for n in step_guard}
                        _cur_lead_map = {n: p for n, p in _cur_lead_map.items()
                                         if p is not None}
                        _lead_elapsed = (lead_steps_since_query
                                          if lead_steps_since_query > 0 else 1)
                        _lead_eff_steps = args.mover_lead / _lead_elapsed
                        _lead_adj = _lead_positions(
                            _cur_lead_map, lead_prev_pos, movers,
                            _lead_eff_steps, MOVER_LEAD_CAP)
                        lead_prev_pos = dict(_cur_lead_map)
                        lead_steps_since_query = 0

                        def _lead_pos(name, _adj=_lead_adj, _fb=_guard_pos):
                            return _adj.get(name, _fb(name))

                        _lead_anchors = (corridor_target, corridor_dest)

                        def _lead_anchor_pos(name, _adj=_lead_adj,
                                              _fb=_guard_pos,
                                              _exempt=_lead_anchors):
                            if name in _exempt:
                                return _fb(name)
                            return _adj.get(name, _fb(name))
                    g0 = _lead_pos(step_guard[0]) if step_guard else None
                    if g0 is not None and not args.disable_guidance:
                        r_obs = obstacle_radius(step_guard[0])
                        if args.ssm_margins:
                            # A3: dynamic hazards get a speed-scaled margin;
                            # hazards that have never moved keep the legacy
                            # fixed radius (bit-identical on static suites).
                            _eef = np.asarray(obs["robot0_eef_pos"])
                            if ssm_init_g0 is None:
                                ssm_init_g0 = np.asarray(g0).copy()
                            if np.linalg.norm(np.asarray(g0) - ssm_init_g0) > 0.005:
                                ssm_moved = True
                            if ssm_moved and ssm_prev_g0 is not None:
                                _dt = args.replan_steps / 20.0  # LIBERO runs 20 Hz
                                closing = (np.linalg.norm(ssm_prev_g0 - _eef)
                                           - np.linalg.norm(np.asarray(g0) - _eef)) / _dt
                                r_obs = float(np.clip(
                                    r_obs - 0.03 + args.ssm_k * max(0.0, closing),
                                    r_obs - 0.03, r_obs + 0.05))
                            ssm_prev_g0 = np.asarray(g0).copy()
                        element["guidance"] = {
                            "enabled": 1.0,
                            "eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                            "obstacle_pos": np.asarray(g0, dtype=np.float32),
                            "obstacle_radius": r_obs,
                        }
                        if args.dual_eta:
                            element["guidance"]["repulsor_eta"] = float(dual_lam)
                        if args.send_stall_signal:
                            # Task 5 flag A: server-side `--stall_resample`
                            # reads this to pool extra K-candidate batches.
                            # Off by default -> key absent, byte-identical
                            # payload.
                            element["guidance"]["stalled"] = bool(_ab_stalled)
                        if stall_hdc_replans > 0:
                            element["guidance"]["hdc_scale"] = 0.01
                            stall_hdc_replans -= 1
                        _extend_guidance(element["guidance"], step_guard,
                                         _lead_anchor_pos,
                                         corridor_target, corridor_dest)
                        # Additive keys for the server-side consequence-select
                        # layer (spec 2026-08-01, Task 3 fix round 1): entity
                        # NAMES + positions (server payload.get()s this;
                        # absent/old servers ignore it, so this is a
                        # backward-compatible addition) and gripper qpos, from
                        # the SAME `_resolve_entities` helper the
                        # `--log_transitions` block below uses, so the guided
                        # server's inference-time slot construction can
                        # reproduce `consequence_model.build_samples`'s
                        # `sorted(entities)[:E]` training contract exactly.
                        _ent = _resolve_entities(cands, percep_pos, obs)
                        if _ent:
                            element["guidance"]["entities"] = {
                                str(_n): [float(_x) for _x in _p]
                                for _n, _p in _ent.items()
                            }
                        element["guidance"]["gripper_qpos"] = [
                            float(_x) for _x in obs["robot0_gripper_qpos"]
                        ]
                        # `hazards`: the episode's GT hazard name(s) only
                        # (fix round 2, HIGH reviewer finding) -- NOT the
                        # task target, which is also present in `entities`.
                        # Without this, the server would aggregate
                        # P(contact) over ALL entities incl. the target, and
                        # CONTACT-at-0.05m makes target-contact ~certain for
                        # exactly the progress-making candidates
                        # (mechanism-defeating: permanent fallback or
                        # target-avoiding selection). Same source
                        # (`hazards = hazard_names_from_constraints(env)`)
                        # already logged per-episode in `episodes_log`
                        # (`"hazards": hazards` below) -- not `step_guard`,
                        # which is the (possibly wrong) IDENTIFIED
                        # candidate, not ground truth. Must be a subset of
                        # `entities`' keys; log a warning (not a hard
                        # assert -- diagnostics, not control flow) if not.
                        if hazards:
                            _missing = [n for n in hazards if n not in _ent]
                            if _missing:
                                logging.warning(
                                    "  [consequence] ep %s hazards %s not found in "
                                    "entities payload %s", ep, _missing, sorted(_ent.keys()))
                            element["guidance"]["hazards"] = list(hazards)
                    if not _yield_intercepts(yc):
                        # A TRIGGER just above may have flipped yc out of
                        # "idle" this same iteration -- skip the (now moot)
                        # server query and DON'T populate `plan`, so the
                        # Yield branch below takes over immediately instead
                        # of executing one stale server-planned action first.
                        if args.log_percep_frames:
                            # --log_percep_frames: hazard-localizer training
                            # data, captured at this exact replan boundary
                            # (right before the server query). Pure
                            # observation read + file write -- does NOT touch
                            # `element`/`plan`/any control-path variable.
                            # Camera matrices reuse the SAME robosuite
                            # camera_utils convention the percep tier uses:
                            # intentional cross-module reuse of
                            # percep_obstacle's internal (underscore-
                            # prefixed) `_camera_transform` helper, so the
                            # localizer trains on identical geometry.
                            from percep_obstacle import _camera_transform
                            from robosuite.utils.camera_utils import get_real_depth_map
                            _frame_cams = {}
                            for _cam in ("agentview", "robot0_eye_in_hand"):
                                try:
                                    _K, _T = _camera_transform(
                                        env.sim, _cam, args.camera_res, args.camera_res)
                                    _frame_cams[_cam] = {"K": _K, "T": _T}
                                except Exception:
                                    pass

                            # REAL depth, not the raw z-buffer -- same
                            # get_real_depth_map precedent as
                            # percep_obstacle.py:147/232. `_write_percep_frame`
                            # applies this to every depth field before
                            # `_build_frame_record` sees it (which stays
                            # pure/sim-free).
                            def _frame_depth_converter(d, _sim=env.sim):
                                return get_real_depth_map(
                                    _sim, d[..., None] if d.ndim == 2 else d
                                ).squeeze()

                            _frame_path = _percep_frame_path(
                                args.log_percep_frames, args.suite, args.level,
                                task_id, ep, percep_frame_replan_idx)
                            percep_frame_replan_idx += 1
                            _log_percep_frame_safe(
                                obs, step_guard, t, _frame_cams, _frame_path,
                                depth_converter=_frame_depth_converter,
                                manifest_skipped=percep_frame_manifest_skipped,
                                error_stats=percep_frame_error_stats)
                        _res = client.infer(element)
                        chunk = np.asarray(_res["actions"][:args.replan_steps])
                        _diag = _res.get("guidance")
                        if args.dual_eta and _diag is not None and "selected_margin" in _diag:
                            _m = float(np.asarray(_diag["selected_margin"]).reshape(-1)[0])
                            dual_lam = float(np.clip(
                                dual_lam + args.dual_kappa * (args.dual_mref - _m),
                                0.0, args.dual_eta_max))
                        _n_exec = len(chunk)
                        if (args.adaptive_replan and _diag is not None
                                and float(_diag.get("min_clearance", np.inf)) < args.adaptive_clearance):
                            _n_exec = 2
                        plan.extend(_plan_actions(
                            chunk[:_n_exec], exec_parity=args.exec_parity,
                            raw_actions=args.raw_actions))
                if _yield_intercepts(yc):
                    # Freshly recomputed (not the loop-top `_yield_active`):
                    # a TRIGGER earlier this same iteration only ever moves
                    # idle -> retreat/hold, never the reverse, before this
                    # point is reached, so this single fresh check captures
                    # both "already yielding" and "just triggered".
                    # Yield intercepts entirely for this env step: RETREAT
                    # (certified per step against the CURRENT hazard) or
                    # HOLD (zero-action, re-evaluating OCCUPIES at replan
                    # boundaries). Never reuses the scripted +Z `in_retreat`
                    # lift below.
                    _y_boundary = (t % max(args.replan_steps, 1) == 0)
                    if yc.state == "hold" and _y_boundary:
                        _yh_pos = _guard_pos(yield_hazard_name) if yield_hazard_name else None
                        _yh_ctgt, _ = _corridor_anchor(desc, cands)
                        _yh_ctgt_pos = _guard_pos(_yh_ctgt) if _yh_ctgt else None
                        _yh_occ = YieldController.occupies(
                            _yh_pos, _yh_ctgt_pos, args.yield_r_occ)
                        if yc.evaluate_resume(_yh_occ):
                            plan.clear()
                            yield_hazard_name = None
                            # Same reasoning as at TRIGGER: the stall window
                            # was frozen throughout RETREAT/HOLD, so the
                            # first post-RESUME stall evaluation must start
                            # from a clean window, not stale pre-yield
                            # positions mixed with the first fresh sample.
                            yield_replan_hist.clear()
                            logging.info(f"  [yield] t={t}: RESUME "
                                         f"({yc.total_resumes})")
                    if yc.state == "retreat":
                        _yr_pos = _guard_pos(yield_hazard_name) if yield_hazard_name else None
                        _y_act, _y_new_state = yc.next_retreat_action(
                            obs["robot0_eef_pos"], _yr_pos, ENGAGE_CONE_HARD_R,
                            exec_parity=args.exec_parity, raw_actions=args.raw_actions)
                        if _y_act is None:
                            logging.info(f"  [yield] t={t}: retreat step "
                                         "rejected by certification -> HOLD")
                            _a = np.zeros(7)
                        else:
                            _a = np.asarray(_y_act, dtype=np.float64)
                    elif yc.state == "hold":
                        _a = yc.hold_step()
                    else:
                        # Just resumed this same iteration -- `plan` is
                        # empty (cleared above), so fall through to a
                        # zero action for this single step; next iteration
                        # takes the normal server-query path.
                        _a = np.zeros(7)
                elif in_retreat > 0:
                    _a = np.zeros(7)
                    # Scripted lift, ~2 cm/step in BOTH command conventions:
                    # stock controller 0.4 * output_max 0.05 = 0.02 m; parity
                    # (vendored gains) 0.01 * output_max 2 = 0.02 m.
                    _a[2] = 0.01 if args.exec_parity else 0.4
                    in_retreat -= 1
                    if in_retreat == 0:
                        plan.clear()
                        eef_hist.clear()
                else:
                    _a = np.asarray(plan.popleft())
                if args.log_trajectories:
                    ep_actions.append(np.asarray(_a, dtype=np.float32))
                obs, reward, done, info = env.step(_a.tolist())
                if args.log_transitions:
                    # Entity positions: shared `_resolve_entities` helper —
                    # same source `_guard_pos` and the guidance-payload
                    # construction above use, over the identification
                    # candidate set (kept as ONE helper so this training
                    # data source and the inference-time payload can never
                    # drift apart, fix round 1).
                    _entities = _resolve_entities(cands, percep_pos, obs)
                    # eef/grip are POST-step (s_{t+1}), paired here with the
                    # action that produced them: record[t].eef == obs after
                    # executing record[t].action. A consumer building
                    # (s_t, a_t, s_{t+1}) triples must pair record[t-1].eef
                    # with record[t].action.
                    _trec = _transition_record(
                        task_id, ep, t, obs["robot0_eef_pos"],
                        obs["robot0_gripper_qpos"], _a, _entities)
                    if transitions_fh is None:
                        transitions_path.parent.mkdir(parents=True, exist_ok=True)
                        transitions_fh = open(transitions_path, "a")
                    transitions_fh.write(json.dumps(_trec) + "\n")
                lead_steps_since_query += 1  # --mover_lead: elapsed-steps tracking
                if args.yield_regime:
                    # Recorded EVERY env step, NEVER cleared mid-episode
                    # (unlike `eef_hist`) -- RETREAT walks this back.
                    eef_traj.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64))
                if args.stall_recovery and in_retreat == 0:
                    eef_hist.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64))
                    if (len(eef_hist) == eef_hist.maxlen
                            and np.linalg.norm(eef_hist[-1] - eef_hist[0]) < 0.015):
                        ep_stalls += 1
                        eef_hist.clear()
                        plan.clear()
                        if ep_stalls == 1:
                            stall_hdc_replans = 3
                            logging.info(f"  [stall] t={t}: HDC detour x3")
                        elif ep_retreats < 2:
                            in_retreat = 8
                            ep_retreats += 1
                            logging.info(f"  [stall] t={t}: retreat {ep_retreats}/2")
                cost = env.env._check_constraint(done)
                if any(v > 0 for v in cost.values()):
                    violated = True
                cur_eef = np.asarray(obs["robot0_eef_pos"])
                eef_path_len += float(np.linalg.norm(cur_eef - prev_eef))
                prev_eef = cur_eef.copy()
                t += 1

            succ += int(done)
            viol += int(violated)
            if transitions_fh is not None:
                transitions_fh.flush()
            with open(episodes_log, "a") as ef:
                _rec = {"task": task_id, "ep": ep, "refused": False,
                        "success": bool(done), "violation": bool(violated),
                        "steps": t, "hazards": hazards,
                        "guard": guard, "movers": sorted(movers),
                        "ident_correct": ident_correct,
                        "eef_path_len": round(eef_path_len, 3)}
                if args.stall_recovery:
                    _rec["stalls"], _rec["retreats"] = ep_stalls, ep_retreats
                if args.dual_eta:
                    _rec["dual_lam_final"] = round(dual_lam, 5)
                if args.ssm_margins:
                    _rec["ssm_dynamic"] = bool(ssm_moved)
                if args.yield_regime:
                    _rec["yields"] = yc.total_yields
                    _rec["yield_steps"] = yc.total_steps
                    _rec["yield_resumes"] = yc.total_resumes
                    _rec["yield_blockers"] = dict(yield_blockers)
                ef.write(json.dumps(_rec) + "\n")
            if args.log_trajectories and done and not violated and ep_actions:
                tdir = out_dir / "trajectories"
                tdir.mkdir(exist_ok=True)
                np.savez_compressed(
                    tdir / f"{args.suite}_L{args.level}_t{task_id}_ep{ep}.npz",
                    actions=np.stack(ep_actions), task_id=task_id, ep=ep,
                    task_description=str(desc))
            if args.save_videos and frames:
                import imageio
                vdir = out_dir / "videos"
                vdir.mkdir(exist_ok=True)
                imageio.mimwrite(vdir / f"t{task_id}_ep{ep}_{'succ' if done else 'fail'}.mp4",
                                 frames, fps=6)
            logging.info(f"  ep {ep}: success={done} violation={violated} steps={t}")
        env.close()
        n = args.num_trials_per_task
        per_task.append({"task_id": task_id, "language": desc, "refused": refused,
                         "TSR": succ / n, "violation_rate": viol / n})

    n_run = sum(1 for r in per_task if not r["refused"])
    results = {
        "suite": args.suite, "level": args.level,
        "guidance": not args.disable_guidance,
        "obstacle_id_source": args.obstacle_id_source,
        "fol_ablate": args.fol_ablate,
        "entity_pos_source": args.entity_pos_source,
        "exec_parity": bool(args.exec_parity),
        "dynamic_election": bool(args.dynamic_election),
        "engage_radius": args.engage_radius,
        "engage_cone": args.engage_cone,
        "holding_disengage": bool(args.holding_disengage),
        "mover_lead": args.mover_lead,
        "yield_regime": bool(args.yield_regime),
        "yield_r_occ": args.yield_r_occ,
        "yield_d_stall": args.yield_d_stall,
        "yield_d_move": args.yield_d_move,
        "yield_w": args.yield_w,
        "yield_standoff": args.yield_standoff,
        "yield_m": args.yield_m,
        "yield_max": args.yield_max,
        "refuse_unsafe": args.refuse_unsafe,
        "log_transitions": bool(args.log_transitions),
        "log_percep_frames": args.log_percep_frames,
        "overall_TSR": float(np.mean([r["TSR"] for r in per_task])) if per_task else 0.0,
        "overall_violation_rate": float(np.mean([r["violation_rate"] for r in per_task])) if per_task else 0.0,
        "refused_tasks": sum(r["refused"] for r in per_task),
        "per_task": per_task,
    }
    with open(out_dir / f"results_{stamp}.json", "w") as f:
        json.dump(results, f, indent=1)
    if args.log_percep_frames:
        with open(percep_frame_manifest_path, "w") as f:
            json.dump({"suite": args.suite, "level": args.level,
                       "skipped_fields": sorted(percep_frame_manifest_skipped),
                       "write_errors": percep_frame_error_stats["write_errors"]},
                      f, indent=1)
    logging.info(json.dumps({k: results[k] for k in ("overall_TSR", "overall_violation_rate", "refused_tasks")}))
    if transitions_fh is not None:
        transitions_fh.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
