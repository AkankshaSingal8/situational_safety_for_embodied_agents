"""In-denoising DCBF safety guidance for the JAX Pi0/pi0.5 model.

Implements the enforcement mechanism of "Neuro-Symbolic Safety Guidance for
VLA Models via Constrained Flow Matching" (arXiv 2607.01378): after every
Euler step of flow-matching denoising, the intermediate action chunk is
interpreted as a future end-effector trajectory (single-integrator rollout of
the translational deltas) and repaired so that a discrete exponential-CBF
condition holds along the horizon:

    B_j >= (1 - gamma) * B_{j-1},   B_j = ||p_j - o|| - r_eff

The repair is a single in-order sweep over the horizon: constraint j depends
only on deltas 0..j, so fixing violations in order yields an exactly feasible
chunk (up to the per-step command clamp) while modifying only violating steps
— a minimum-norm-flavoured correction without an external solver. Everything
is jnp and shape-static, so the whole sampler jits.

This module is additive: `guided_sample_actions` is bound to an existing Pi0
instance with `types.MethodType`; nothing in upstream openpi is modified.
"""

from typing import NamedTuple

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.shared import array_typing as at


class GuidanceParams(NamedTuple):
    """Per-request guidance inputs. All jnp arrays -> traced, no recompiles.

    Metric quantities come straight from the simulator via the client, so no
    state denormalization is needed inside the model. Action denormalization
    constants (quantile stats for the translational dims) are baked in by the
    policy wrapper from the checkpoint's norm stats.
    """

    enabled: at.Float[at.Array, ""]  # 1.0 = apply correction, 0.0 = passthrough
    eef_pos: at.Float[at.Array, "3"]  # current metric EEF position p_0
    obstacle_pos: at.Float[at.Array, "3"]  # metric obstacle center o
    r_eff: at.Float[at.Array, ""]  # obstacle radius + eef radius + d_safe
    inflation_slope: at.Float[at.Array, ""]  # extra margin per horizon step [m] (open-loop tracking uncertainty)
    gamma: at.Float[at.Array, ""]  # DCBF decay rate (0.9 = SOTA setting)
    q01: at.Float[at.Array, "3"]  # action quantile stats, translational dims
    q99: at.Float[at.Array, "3"]
    translation_scale: at.Float[at.Array, ""]  # metres of EEF motion per unit command
    # Per-step clamp in COMMAND units. 1.0 = the [-1,1] command convention
    # (every SafeLIBERO row, bit-exact). Metric-delta checkpoints (LIBERO-
    # Safety finetune: actions unnormalize to meters) serve with
    # translation_scale=1.0 and cmd_clip=0.05 so the repair's feasibility
    # model stays "at most 5 cm of EEF motion per control step".
    cmd_clip: at.Float[at.Array, ""]
    # Denoising-time schedule (length = num denoising steps). repair_weight
    # scales the correction applied at Euler step k (0 = trust the flow, 1 =
    # full repair); margin_scale scales r_eff at step k. Rationale: always-hard
    # repair from pure noise is the documented SafeDiffuser-RoS trap pattern —
    # trust early, correct softly mid, enforce exactly late. The LAST entry of
    # both arrays must be 1.0 so the terminal chunk satisfies the exact chain.
    repair_weight: at.Float[at.Array, "n"]
    margin_scale: at.Float[at.Array, "n"]
    # Semantic corridor exemption: margins RELAX (never vanish) for rollout
    # points inside a cylinder around the sanctioned approach segments
    # eef->target and eef->destination. This is what lets fat protective
    # envelopes coexist with grasping/placing next to an obstacle — a uniform
    # envelope cannot distinguish sanctioned approach from hazard approach.
    # Disable by placing target/dest far away (the client's default).
    target_pos: at.Float[at.Array, "3"]
    dest_pos: at.Float[at.Array, "3"]
    corridor_radius: at.Float[at.Array, ""]  # cylinder radius [m]
    corridor_relax: at.Float[at.Array, ""]  # margin multiplier inside = (1 - relax)
    companion_scale: at.Float[at.Array, ""]  # 1 = hand/payload points on; 0 = EEF-point-only (SOTA-reimpl fidelity)
    # DBNR (Damage-Budgeted Nullspace Restitution, docs/research/inv_invB.json
    # methods[1]): on a step the sweep just corrected, restore task-aligned
    # velocity in the exact nullspace of THIS step's push direction, budgeted
    # proportionally to the correction just applied. Single-obstacle
    # simplification of the general spec (no cross-step debt, no multi-face
    # slack cap — this repair has at most one active constraint per step, so
    # the nullspace projector is exactly I - dd^T with no other faces to
    # protect). 0 = off (default; matches every existing config/ablation row).
    dbnr_beta0: at.Float[at.Array, ""]
    # Anisotropic obstacle shape (superquadric): semi-axes [m] of the
    # obstacle's own extent. All-zero = sphere mode (r_eff as-is, exact legacy
    # behavior). When set, the barrier's obstacle radius becomes
    # direction-dependent — r(u) = (sum_i |u_i/s_i|^(2/eps))^(-eps/2), the
    # radial distance from center to the superquadric surface along u — so a
    # tall thin moka pot gets a tall thin keep-out instead of one scalar
    # radius forced to trade TSR (fat) against CAR (thin). r_obs_base is the
    # scalar radius the client folded into r_eff; it is added back so
    # b = dist - r_shape(u) - eef_radius - d_safe.
    obstacle_scales: at.Float[at.Array, "3"]
    sq_eps: at.Float[at.Array, ""]  # superquadric exponent (1 = ellipsoid, ->0 = box)
    r_obs_base: at.Float[at.Array, ""]
    # HDC (Homotopy-Diverse Candidates, DBNR successor — see ledger 2026-07-16):
    # bias a fixed subset of the K noise candidates with a lateral (left/right
    # around the obstacle) per-step velocity bias during mid denoising steps,
    # so executed-prefix acceptance can SELECT a different route homotopy on
    # obstacle-on-path scenes instead of repairing inside the blocked one.
    # Value = desired lateral displacement in meters PER ACTION STEP for the
    # biased candidates (0 = off; K=1 requests are never biased).
    hdc_scale: at.Float[at.Array, ""]
    # Guard-set second obstacle (fixed M=2 keeps jit shapes static). Default =
    # FAR center + zero scales -> its barrier is huge -> min() returns the
    # primary barrier bit-for-bit. extra_r_obs is obstacle 2's OWN scalar
    # radius; its effective radius reuses the primary's eef_radius + d_safe
    # via r_eff2 = extra_r_obs + (r_eff - r_obs_base).
    extra_obstacle_pos: at.Float[at.Array, "3"]
    extra_r_obs: at.Float[at.Array, ""]
    extra_obstacle_scales: at.Float[at.Array, "3"]
    # Enforcement pilot knobs (both 0.0 = machinery inert, bit-exact legacy).
    # fk_beta: FK/SMC tilt strength — candidates reweighted by
    # softmax(beta * prefix_margin) and systematically resampled at two
    # mid-denoising steps (candidate B). Uniform weights resample to identity.
    fk_beta: at.Float[at.Array, ""]
    # repulsor_eta: soft velocity-superposition repulsor strength [m per
    # denoise step at full hinge] added to translational commands (candidate C).
    repulsor_eta: at.Float[at.Array, ""]
    # Revision knobs (spec 2026-07-27; both 0.0 = bit-exact legacy selection).
    # progress_weight: weight on DIRECTED prefix displacement toward target_pos
    # in best-of-K selection (A1) — replaces "any motion" with "motion that
    # advances the task" among certified-safe candidates.
    progress_weight: at.Float[at.Array, ""]
    # lookahead_weight: weight on the post-prefix (tail) minimum barrier margin
    # (C) — candidates whose tail re-enters the keep-out lose ties, avoiding
    # geometric dead-ends that stall the next replan.
    lookahead_weight: at.Float[at.Array, ""]
    # tilt_lambda: DERIVED steering operator (constrained-sampling formulation,
    # 2026-07-28). The safety-tilted target distribution is
    # pi(chunk) * exp(-lambda * danger(chunk)); the correct guidance drift is
    # -lambda * grad_x danger, where danger = sum of hinge barrier violations
    # along the analytic rollout (same margins as the acceptance certificate,
    # corridor exemption INCLUDED — so "don't fight the sanctioned approach"
    # emerges from the derivation instead of a bolted-on gate). The gradient
    # direction is normalized per candidate and scaled by tilt_lambda in the
    # SAME meters-per-denoise-step units as repulsor_eta, so the client's dual
    # (lambda) controller transfers unchanged. 0 = inert, bit-exact legacy.
    tilt_lambda: at.Float[at.Array, ""]


def _eff_dist(diffs: jnp.ndarray, g: GuidanceParams, scales=None, r_base=None, shape_scale=1.0) -> jnp.ndarray:
    """Shape-corrected distance for center offsets `diffs` (..., 3).

    Sphere mode (any obstacle_scale <= 0): plain Euclidean norm — callers'
    `dist - r_eff` reproduces legacy behavior bit-for-bit. Shape mode
    (ellipsoid semi-axes in `scales`): support-plane true-distance bound, so
    with shape_scale=1 the caller's `dist_eff - r_eff` equals
    dist(p, ellipsoid) - eef_radius - d_safe (conservative; C2-correct —
    the pre-2026-07-30 radial form under-enforced obliquely).
    shape_scale < 1 (corridor/margin relaxation) shrinks the anisotropic
    keep-out the same way it shrinks the scalar margin — without it the
    corridor exemption cannot reach the shape term at all (the t1 failure).
    scales/r_base default to the primary obstacle's; pass the extra
    obstacle's to evaluate its shape.
    """
    scales = g.obstacle_scales if scales is None else scales
    r_base = g.r_obs_base if r_base is None else r_base
    dists = jnp.linalg.norm(diffs, axis=-1)
    s = jnp.maximum(scales, 1e-3)
    # Ellipsoid SUPPORT-PLANE distance (2026-07-30 C2 fix). The old radial
    # form subtracted r_shape(u) — the gap along the center ray — which
    # OVERSTATES the true clearance obliquely, under-enforcing the margin by
    # up to 69 mm (audit C2). The support-plane bound d = (p-c)·n - h_E(n)
    # with n ∝ diag(1/s²)(p-c) is a LOWER bound on dist(p, E) for any convex
    # E, so enforcing it is conservative — errors now widen margins instead
    # of silently shrinking them. Exact on-axis and for spheres.
    n_raw = diffs / (s * s)
    n_hat = n_raw / (jnp.linalg.norm(n_raw, axis=-1, keepdims=True) + 1e-8)
    support = jnp.sqrt(jnp.sum((s * n_hat) ** 2, axis=-1) + 1e-12)
    d_true = jnp.sum(diffs * n_hat, axis=-1) - support  # <= dist(p, E); < 0 inside
    # Same relaxation algebra as before: r_eq plays the old r_shape role, so
    # shape_scale (corridor/margin relaxation) and the caller's `- r_eff`
    # compose identically; at shape_scale=1, pseudo - r_eff = d_true - pad.
    r_eq = dists - d_true
    use_shape = jnp.min(scales) > 0.0
    return jnp.where(use_shape, dists - shape_scale * (r_eq - r_base), dists)


def _min_eff_dist(p: jnp.ndarray, g: GuidanceParams, shape_scale=1.0) -> jnp.ndarray:
    """Guard-set pseudo-distance at points p (..., 3): min over the two
    obstacles of their barrier values, re-offset by the PRIMARY r_eff so that
    callers' existing `- g.r_eff` yields the min barrier. Obstacle 2 at FAR
    (the client default) makes this exactly the primary-only value."""
    b1 = _eff_dist(p - g.obstacle_pos, g, shape_scale=shape_scale) - g.r_eff
    r_eff2 = g.extra_r_obs + (g.r_eff - g.r_obs_base)
    b2 = _eff_dist(p - g.extra_obstacle_pos, g, g.extra_obstacle_scales,
                   g.extra_r_obs, shape_scale=shape_scale) - r_eff2
    return jnp.minimum(b1, b2) + g.r_eff


def _corridor_scale(p: jnp.ndarray, g: GuidanceParams) -> jnp.ndarray:
    """Margin multiplier at point(s) p: 1 outside sanctioned corridors,
    (1 - corridor_relax) inside the cylinder around eef->target or eef->dest.

    p: (..., 3). Returns (...,). Segments shorter than 2 cm are ignored (the
    far-away disable convention makes them ~17 m long instead).
    """

    def seg_dist(q):
        v = q - g.eef_pos
        vv = jnp.maximum(jnp.sum(v * v), 1e-8)
        t_raw = jnp.einsum("...i,i->...", p - g.eef_pos, v) / vv
        t = jnp.clip(t_raw, 0.0, 1.0)
        proj = g.eef_pos + t[..., None] * v
        d = jnp.linalg.norm(p - proj, axis=-1)
        # Two validity gates: (a) the entity must be within workspace scale
        # (the far-away disable convention), and (b) the point must lie a
        # genuine fraction ALONG the segment — otherwise clamping at t=0 would
        # turn the corridor into an exemption sphere around the EEF itself,
        # relaxing protection in every direction including toward the hazard.
        valid = (jnp.linalg.norm(q - g.eef_pos) < 2.0) & (t_raw > 0.15)
        return jnp.where(valid, d, jnp.inf)

    d_min = jnp.minimum(seg_dist(g.target_pos), seg_dist(g.dest_pos))
    inside = (d_min < g.corridor_radius).astype(jnp.float32)
    return 1.0 - g.corridor_relax * inside


def _dcbf_repair(x_t: jnp.ndarray, g: GuidanceParams, step_idx) -> tuple[jnp.ndarray, dict]:
    """Repair the (B, H, action_dim) normalized chunk to satisfy the DCBF chain.

    Only dims 0:3 (translational commands) are modified. Returns the repaired
    chunk and diagnostics. The correction is scaled by `g.enabled` and by the
    denoising-time schedule `g.repair_weight[step_idx]` so a single jitted code
    path serves guided, unguided, and scheduled requests.

    Repair channel is brake-then-push (PC-Diffuser insight): the correction is
    applied along the radial escape direction, and if the |command|<=1 clamp
    cannot realize the required push, the step falls back to a FULL BRAKE
    (remove all motion toward the obstacle), which guarantees the barrier is
    non-decreasing at that step — chain feasibility degrades gracefully to
    non-worsening under actuation limits instead of executing a known-violating
    step.
    """
    b, h, _ = x_t.shape
    w_k = g.repair_weight[step_idx]
    ms_k = g.margin_scale[step_idx]

    # normalized -> env command in [-1, 1] (quantile unnormalize, cf.
    # transforms.Unnormalize._unnormalize_quantile)
    span = g.q99 - g.q01 + 1e-6
    cmd = (x_t[..., :3] + 1.0) / 2.0 * span + g.q01  # (B, H, 3)
    disp = cmd * g.translation_scale  # metric per-step displacement

    # Vertical companion points approximating the full manipulated system, not
    # just the EEF frame origin: the hand/wrist body sits above the grip site
    # (v19 forensics: unmonitored hand plowed obstacles), fingers and a carried
    # object hang below. Each shares r_eff; the barrier is the min over points.
    COMPANIONS = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.08], [0.0, 0.0, -0.06]]) * g.companion_scale

    def _closest(p, shape_scale=1.0):
        """Min pseudo-distance (and escape direction) over companion points
        AND the two guard-set obstacles; escape direction = center ray of the
        argmin (obstacle, companion) pair.

        The escape direction stays radial: in shape mode moving along u leaves
        r_shape(u) fixed while dist grows, so d_eff increases monotonically —
        the same push/brake monotonicity argument as the sphere.
        """
        pts = p[None, :] + COMPANIONS  # (3, 3)
        diffs1 = pts - g.obstacle_pos
        b1 = _eff_dist(diffs1, g, shape_scale=shape_scale) - g.r_eff
        r_eff2 = g.extra_r_obs + (g.r_eff - g.r_obs_base)
        diffs2 = pts - g.extra_obstacle_pos
        b2 = _eff_dist(diffs2, g, g.extra_obstacle_scales, g.extra_r_obs,
                       shape_scale=shape_scale) - r_eff2
        b_all = jnp.concatenate([b1, b2])  # (6,)
        diffs_all = jnp.concatenate([diffs1, diffs2], axis=0)
        k = jnp.argmin(b_all)
        d = diffs_all[k]
        return b_all[k] + g.r_eff, d / (jnp.linalg.norm(d) + 1e-8)

    # DBNR task direction: toward dest_pos, falling back to target_pos, reused
    # from the corridor module's FAR-disable convention. Loop-invariant (goal
    # doesn't move within a chunk), computed once.
    valid_dest = jnp.linalg.norm(g.dest_pos - g.eef_pos) < 2.0
    valid_target = jnp.linalg.norm(g.target_pos - g.eef_pos) < 2.0
    dbnr_goal = jnp.where(valid_dest, g.dest_pos, jnp.where(valid_target, g.target_pos, g.eef_pos))
    dbnr_have_goal = valid_dest | valid_target

    def repair_one(disp_b):
        """Sweep j = 0..H-1 enforcing B_j >= (1-gamma) * B_{j-1}."""
        p0 = g.eef_pos
        d0, _ = _closest(p0)
        b0 = d0 - g.r_eff

        def body(j, carry):
            disp_acc, p_prev, b_prev = carry
            orig_step = disp_acc[j]
            p_j = p_prev + orig_step
            c_j = _corridor_scale(p_j, g)
            r_j = ms_k * (g.r_eff + g.inflation_slope * (j + 1.0)) * c_j
            # The same relaxation multiplier must reach the anisotropic shape
            # term, or the corridor exempts centimeters of scalar margin while
            # the AABB-sized keep-out still covers the grasp (t1 failure).
            ss_j = ms_k * c_j
            dist, direction = _closest(p_j, ss_j)
            b_j = dist - r_j
            need = (1.0 - g.gamma) * b_prev
            deficit = jnp.maximum(need - b_j, 0.0)
            # Push p_j (and, through the cumulative rollout, all later points)
            # away from the obstacle along the closest companion's direction.
            pushed = orig_step + deficit * direction
            pushed_cmd = jnp.clip(pushed / g.translation_scale, -g.cmd_clip, g.cmd_clip)
            pushed = pushed_cmd * g.translation_scale
            p_push = p_prev + pushed
            dist_push, _ = _closest(p_push, ss_j)
            # FULL-BRAKE alternative: remove the toward-obstacle component of
            # the original step. Its radial component is non-negative, so the
            # barrier cannot decrease at this step — a clamp-independent floor.
            approach = jnp.maximum(jnp.dot(orig_step, -direction), 0.0)
            braked = orig_step + approach * direction
            braked_cmd = jnp.clip(braked / g.translation_scale, -g.cmd_clip, g.cmd_clip)
            braked = braked_cmd * g.translation_scale
            dist_brake, _ = _closest(p_prev + braked, ss_j)
            # Take whichever realizable step yields the larger barrier: the
            # clamped push when it helps (incl. max-speed escape from inside
            # the margin), the brake floor when per-dim clamping would make the
            # push worse than simply not approaching.
            better = jnp.where(dist_push >= dist_brake, pushed, braked)
            # Only correct violating steps: deficit == 0 -> keep the original.
            chosen = jnp.where(deficit > 0.0, better, orig_step)
            # Denoising-time schedule: scale the CORRECTION, not the step.
            new_step = orig_step + w_k * (chosen - orig_step)

            # DBNR restitution: restore task-aligned velocity in the exact
            # nullspace of THIS step's push direction, budgeted proportionally
            # to the damage the repair just did. w_perp is provably orthogonal
            # to `direction` (dot(w_perp, direction) == 0 by construction), so
            # adding gamma * w_perp cannot reduce dist(p_j, obstacle) along the
            # repair's own escape axis — the certified margin from `chosen` is
            # preserved BEFORE the final actuator clamp (property (i) of the
            # spec, single-constraint case). The clamp below can reintroduce a
            # small component along `direction` under saturation; that residual
            # is caught by next step's repair exactly like any other
            # linearization error in this sweep (same honesty contract as the
            # base repair, per the spec's own caveat).
            p_j_repaired = p_prev + new_step
            w_vec = dbnr_goal - p_j_repaired
            w_hat = w_vec / (jnp.linalg.norm(w_vec) + 1e-8)
            w_perp = w_hat - jnp.dot(w_hat, direction) * direction
            damage = jnp.linalg.norm(w_k * (chosen - orig_step))
            do_restitute = (deficit > 0.0) & dbnr_have_goal & (g.dbnr_beta0 > 0.0)
            restitution = g.dbnr_beta0 * damage * w_perp
            new_step_restituted = new_step + restitution
            new_step_restituted_cmd = jnp.clip(new_step_restituted / g.translation_scale, -g.cmd_clip, g.cmd_clip)
            new_step_restituted = new_step_restituted_cmd * g.translation_scale
            # Gate the WHOLE modification (not just restitution -> 0) on
            # do_restitute so dbnr_beta0=0 (every pre-DBNR config/ablation
            # row) reproduces the exact prior new_step bit-for-bit — the prior
            # code never re-clamped new_step, only its `pushed`/`braked`
            # candidates, so an unconditional clip here would silently change
            # behavior for every existing row even with restitution == 0.
            new_step = jnp.where(do_restitute, new_step_restituted, new_step)

            disp_acc = disp_acc.at[j].set(new_step)
            p_j = p_prev + new_step
            dist2, _ = _closest(p_j, ss_j)
            b_j = dist2 - r_j
            return disp_acc, p_j, b_j

        disp_out, _, _ = jax.lax.fori_loop(0, h, body, (disp_b, p0, b0))
        return disp_out

    disp_repaired = jax.vmap(repair_one)(disp)
    disp_new = g.enabled * disp_repaired + (1.0 - g.enabled) * disp

    # metric -> normalized, write back translational dims only
    cmd_new = disp_new / g.translation_scale
    xt_trans = (cmd_new - g.q01) / span * 2.0 - 1.0
    x_new = x_t.at[..., :3].set(xt_trans)

    # Diagnostics (computed on the possibly-repaired chunk, inflated radii,
    # min over companion points).
    p_traj = g.eef_pos + jnp.cumsum(disp_new, axis=1)  # (B, H, 3)
    r_horizon = g.r_eff + g.inflation_slope * jnp.arange(1.0, h + 1.0)
    comp_dists = _min_eff_dist(
        p_traj[:, :, None, :] + COMPANIONS[None, None, :, :], g
    )  # (B, H, 3)
    clearance = jnp.min(comp_dists, axis=-1) - r_horizon
    b0_val = jnp.min(_min_eff_dist(g.eef_pos[None, :] + COMPANIONS, g)) - g.r_eff
    b_chain = jnp.concatenate([jnp.broadcast_to(b0_val, (b, 1)), clearance], axis=1)
    residual = jnp.maximum((1.0 - g.gamma) * b_chain[:, :-1] - b_chain[:, 1:], 0.0)
    diag = {
        "correction_norm": jnp.linalg.norm(x_new - x_t, axis=(-2, -1)),  # (B,)
        "min_clearance": jnp.min(clearance, axis=-1),  # (B,)
        "max_residual": jnp.max(residual, axis=-1),  # (B,)
    }
    diag.update(_dbnr_diagnostics(disp, disp_new, p_traj, g))
    return x_new, diag


def _dbnr_diagnostics(disp: jnp.ndarray, disp_new: jnp.ndarray, p_traj: jnp.ndarray, g: GuidanceParams) -> dict:
    """DBNR kill-test instrumentation (read-only; does not affect x_new).

    Estimates, per step, how much task-aligned velocity the repair just
    removed (tau) and how much headroom exists to restore it without
    touching the active constraint (headroom = the component of the unit
    task direction orthogonal to the repair's push direction, i.e.
    ||(I - dd^T) w_hat|| for push direction d). This is the single-obstacle
    approximation of the general nullspace-projector story in the DBNR spec
    (docs/research/inv_invB.json methods[1]): with one active constraint per
    step the projector collapses to I - d d^T, no Cholesky/stack needed.

    Kill criterion (pre-registered, CONTEXT_HANDOFF.md #4.5): if tau ~= 0
    across active steps, the repair isn't removing meaningful task motion
    and DBNR has nothing to restore -> kill. If tau > 0 and headroom > 0,
    there is recoverable task progress -> implement the full restitution.
    """
    correction = disp_new - disp  # (B, H, 3) — what the repair actually changed
    corr_norm = jnp.linalg.norm(correction, axis=-1)  # (B, H)
    active = corr_norm > 1e-5

    p_prev = jnp.concatenate([jnp.broadcast_to(g.eef_pos, p_traj[:, :1, :].shape), p_traj[:, :-1, :]], axis=1)

    valid_dest = jnp.linalg.norm(g.dest_pos - g.eef_pos) < 2.0
    valid_target = jnp.linalg.norm(g.target_pos - g.eef_pos) < 2.0
    goal = jnp.where(valid_dest, g.dest_pos, jnp.where(valid_target, g.target_pos, g.eef_pos))
    have_goal = valid_dest | valid_target

    w = goal[None, None, :] - p_prev  # (B, H, 3)
    w_hat = w / (jnp.linalg.norm(w, axis=-1, keepdims=True) + 1e-8)

    # tau: task-aligned velocity the correction removed (positive = opposed task direction)
    tau = jnp.maximum(0.0, -jnp.sum(correction * w_hat, axis=-1))  # (B, H)

    # push direction actually used by the repair, recovered post-hoc from the
    # geometry (companion offsets omitted here — a coarse but adequate proxy
    # for a routing-level diagnostic): points away from the obstacle.
    push_dir = p_traj - g.obstacle_pos[None, None, :]
    push_dir = push_dir / (jnp.linalg.norm(push_dir, axis=-1, keepdims=True) + 1e-8)
    headroom = jnp.linalg.norm(w_hat - jnp.sum(w_hat * push_dir, axis=-1, keepdims=True) * push_dir, axis=-1)  # (B, H)

    mask = active & have_goal
    n_active = jnp.maximum(jnp.sum(mask.astype(jnp.float32), axis=-1), 1.0)  # (B,)
    tau_mean = jnp.sum(jnp.where(mask, tau, 0.0), axis=-1) / n_active
    tau_max = jnp.max(jnp.where(mask, tau, 0.0), axis=-1)
    headroom_mean = jnp.sum(jnp.where(mask, headroom, 0.0), axis=-1) / n_active
    active_frac = jnp.mean(active.astype(jnp.float32), axis=-1)

    return {
        "dbnr_tau_mean": tau_mean,  # (B,)
        "dbnr_tau_max": tau_max,  # (B,)
        "dbnr_headroom_mean": headroom_mean,  # (B,), in [0, 1]
        "dbnr_active_frac": active_frac,  # (B,)
    }


def _prefix_acceptance(x_0: jnp.ndarray, g: GuidanceParams, prefix_len: int) -> dict:
    """Score each candidate chunk by its EXECUTED prefix (first `prefix_len`
    steps — what the receding-horizon client actually runs).

    Returns per-candidate: feasibility of the DCBF chain over the prefix at
    full margins, the minimum prefix clearance margin, and total displacement
    (a mild progress proxy so selection doesn't reward freezing).
    """
    span = g.q99 - g.q01 + 1e-6
    cmd = (x_0[..., :3] + 1.0) / 2.0 * span + g.q01
    disp = cmd * g.translation_scale  # (K, H, 3)
    companions = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.08], [0.0, 0.0, -0.06]]) * g.companion_scale
    p_traj = g.eef_pos + jnp.cumsum(disp, axis=1)  # (K, H, 3)
    # Corridor relaxation must reach the SHAPE term here too, not just the
    # scalar margin — otherwise the certificate vetoes grasp chunks the
    # repair path would allow (the t1 superquadric failure, root-caused
    # 2026-07-28: handle-inflated AABB puts the grasp target inside the
    # unrelaxed anisotropic keep-out). Sphere mode: bit-exact no-op.
    cs = _corridor_scale(p_traj, g)  # (K, H)
    dists = jnp.min(
        _min_eff_dist(p_traj[:, :, None, :] + companions[None, None], g,
                      shape_scale=cs[:, :, None]), axis=-1
    )  # (K, H)
    h = disp.shape[1]
    r_horizon = (g.r_eff + g.inflation_slope * jnp.arange(1.0, h + 1.0)) * cs
    b = dists - r_horizon  # (K, H)
    b0 = jnp.min(_min_eff_dist(g.eef_pos[None, :] + companions, g)) - g.r_eff
    chain = jnp.concatenate([jnp.broadcast_to(b0, (b.shape[0], 1)), b[:, :prefix_len]], axis=1)
    viol = (1.0 - g.gamma) * chain[:, :-1] - chain[:, 1:]
    feasible = jnp.max(viol, axis=1) <= 1e-4  # (K,)
    min_margin = jnp.min(b[:, :prefix_len], axis=1)
    net = jnp.sum(disp[:, :prefix_len], axis=1)
    disp_norm = jnp.linalg.norm(net, axis=-1)
    # A1 directed progress: prefix displacement projected onto the unit vector
    # toward target_pos. Falls back to disp_norm under the far-away disable
    # convention (target ~17 m out) so progress_weight can stay on everywhere.
    to_tgt = g.target_pos - g.eef_pos
    tnorm = jnp.linalg.norm(to_tgt)
    directed = jnp.einsum("ki,i->k", net, to_tgt / (tnorm + 1e-8))
    progress = jnp.where(tnorm < 5.0, directed, disp_norm)
    # C tail margin: worst clearance over the post-prefix horizon (dead-end
    # signal). Horizon == prefix would leave an empty slice; reuse the last
    # prefix step in that degenerate case.
    tail = b[:, prefix_len:] if b.shape[1] > prefix_len else b[:, prefix_len - 1:]
    tail_margin = jnp.min(tail, axis=1)
    return {"feasible": feasible, "min_margin": min_margin, "disp_norm": disp_norm,
            "progress": progress, "tail_margin": tail_margin, "net_disp": net,
            # Full per-step metric world-frame displacement (K, H, 3) — added
            # for the consequence-select layer (openpi_guided/src/openpi/
            # policies/guided_policy.py), which needs a per-candidate action
            # window rather than just the summed prefix displacement. Purely
            # additive key; existing consumers only read the keys above.
            "disp": disp}


def _hdc_bias(g: GuidanceParams, k_batch: int) -> jnp.ndarray:
    """HDC per-candidate normalized-action bias, shape (K, 3).

    Lateral unit vector around the obstacle in the table plane; candidate side
    pattern [none, left, right, none] repeating, so K=8 keeps 4 unbiased
    seeds (and K=1 is never biased). The returned delta, summed over the
    denoising window's unit weight, equals `hdc_scale` meters of lateral
    displacement per action step. The repair runs AFTER the bias each step,
    so biased candidates stay margin-certified.
    """
    to_obs = g.obstacle_pos - g.eef_pos
    lat = jnp.cross(to_obs, jnp.array([0.0, 0.0, 1.0]))
    lat = lat / (jnp.linalg.norm(lat) + 1e-8)
    sides = jnp.tile(jnp.array([0.0, 1.0, -1.0, 0.0]), (k_batch + 3) // 4)[:k_batch]
    span3 = g.q99 - g.q01 + 1e-6
    return (
        2.0 * (sides[:, None] * lat[None, :]) * g.hdc_scale
        / (span3[None, :] * g.translation_scale)
    )


def _repulsor(x_t: jnp.ndarray, g: GuidanceParams) -> jnp.ndarray:
    """Enforcement candidate C: soft velocity-superposition repulsor.

    Adds eta * hinge(clearance) along the radial escape direction from the
    PRIMARY obstacle to the translational commands — annealed-free, no
    feasibility logic, no brake: the classic guidance-style baseline. Hinge
    ramps in below TAU_C clearance. eta = 0 -> identity (bit-exact legacy).
    Step-coupling (nudging step j moves p_{j'>j}) is deliberately ignored —
    that is what separates this soft arm from the certified repair."""
    TAU_C = 0.05
    span = g.q99 - g.q01 + 1e-6
    cmd = (x_t[..., :3] + 1.0) / 2.0 * span + g.q01
    disp = cmd * g.translation_scale
    p = g.eef_pos + jnp.cumsum(disp, axis=1)  # (K, H, 3)
    diffs = p - g.obstacle_pos
    dist = jnp.linalg.norm(diffs, axis=-1, keepdims=True)
    hinge = jnp.maximum(0.0, 1.0 - (dist - g.r_eff) / TAU_C)
    ddisp = g.repulsor_eta * hinge * diffs / (dist + 1e-8)
    dx = 2.0 * ddisp / (g.translation_scale * span)
    return x_t.at[..., :3].add(g.enabled * dx)


def _tilt_guidance(x_t: jnp.ndarray, g: GuidanceParams) -> jnp.ndarray:
    """Derived steering: guidance drift of the safety-tilted distribution.

    danger(x) = sum_k relu(TAU_T - b_k(x)) over the FULL rollout horizon,
    with b_k built EXACTLY like the acceptance certificate (companion points,
    horizon-inflated margins, corridor relaxation). The applied drift is the
    per-candidate NORMALIZED negative gradient scaled by tilt_lambda
    [m/denoise-step] — steepest descent on total violation with a controlled
    step, zero wherever the hinge is inactive (safe chunks are untouched).
    Replaces the hand-designed radial repulsor (kept as its ablation)."""
    TAU_T = 0.02  # hinge margin [m]: start steering slightly before violation

    def danger(x3):
        span = g.q99 - g.q01 + 1e-6
        cmd = (x3 + 1.0) / 2.0 * span + g.q01
        disp = cmd * g.translation_scale
        companions = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.08],
                                [0.0, 0.0, -0.06]]) * g.companion_scale
        p_traj = g.eef_pos + jnp.cumsum(disp, axis=1)  # (K, H, 3)
        cs = _corridor_scale(p_traj, g)  # corridor reaches the shape term (t1 fix)
        dists = jnp.min(
            _min_eff_dist(p_traj[:, :, None, :] + companions[None, None], g,
                          shape_scale=cs[:, :, None]), axis=-1
        )
        h = disp.shape[1]
        r_h = (g.r_eff + g.inflation_slope * jnp.arange(1.0, h + 1.0)) * cs
        b = dists - r_h  # (K, H)
        return jnp.sum(jax.nn.relu(TAU_T - b))

    grad = jax.grad(danger)(x_t[..., :3])  # (K, H, 3)
    gnorm = jnp.sqrt(jnp.sum(grad ** 2, axis=(1, 2), keepdims=True))
    direction = grad / (gnorm + 1e-8)
    span = g.q99 - g.q01 + 1e-6
    dx = -2.0 * g.tilt_lambda * direction / (g.translation_scale * span)
    active = (gnorm > 1e-8).astype(jnp.float32)
    return x_t.at[..., :3].add(g.enabled * active * dx)


def _fk_resample(x_t: jnp.ndarray, g: GuidanceParams, k) -> jnp.ndarray:
    """Enforcement candidate B: Feynman-Kac tilt over the K particles.

    At mid-denoising steps k in {4, 7}: weight particles by
    softmax(fk_beta * prefix_margin(x_t)) and systematically resample
    (deterministic offsets — uniform weights map to the IDENTITY permutation,
    so fk_beta = 0 is bit-exact legacy; this is the fidelity check).
    The margin on a partially-denoised x_t is a heuristic score, same
    convention as the repair path's margin_scale."""
    kk = x_t.shape[0]
    m = _prefix_acceptance(x_t, g, 5)["min_margin"]  # (K,)
    w = jax.nn.softmax(g.fk_beta * m)
    cum = jnp.cumsum(w)
    pts = (jnp.arange(kk, dtype=jnp.float32) + 0.5) / kk
    idx = jnp.clip(jnp.searchsorted(cum, pts), 0, kk - 1)
    active = (g.enabled > 0) & (g.fk_beta > 0) & ((k == 4) | (k == 7))
    take = jnp.where(active, idx, jnp.arange(kk))
    return x_t[take]


def guided_sample_actions(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    guidance: GuidanceParams,
    num_steps: int = 10,
    num_candidates: int = 1,
    prefix_len: int = 5,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
    return_candidates: bool = False,
) -> tuple[_model.Actions, dict] | tuple[_model.Actions, dict, _model.Actions, dict, dict]:
    """`Pi0.sample_actions` with a DCBF repair after every Euler step.

    Bind to a loaded Pi0 instance via `types.MethodType(guided_sample_actions,
    model)`. The denoising math replicates upstream `sample_actions` exactly
    (KV-cache prefix, suffix attention, Euler update); the only addition is the
    `_dcbf_repair` call inside the loop, so subsequent denoising steps adapt to
    the correction — the property that distinguishes in-generation guidance
    from post-hoc filtering.

    `return_candidates` (static, default False): when True and `k_batch > 1`,
    additionally returns the full pre-collapse candidate set
    `(selected, diag, x_0, acc, last_diag)` so a host-level layer
    (`consequence_select`, spec 2026-08-01) can override which candidate is
    selected BEFORE this function's own argmax collapse — `selected`/`diag`
    are still the existing legacy selection, untouched, so a caller that
    finds no override applicable can return them verbatim as an exact
    backstop. Default False preserves the original 2-tuple return exactly
    (byte-identical off path).
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    obs_batch = observation.state.shape[0]
    if obs_batch != 1 and num_candidates > 1:
        raise ValueError("num_candidates > 1 requires a single-observation request.")
    k_batch = num_candidates if noise is None else noise.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (k_batch, self.action_horizon, self.action_dim))

    # Prefix (vision + language) computed ONCE at the observation's batch size,
    # then broadcast across candidates: the K suffix passes share one KV cache.
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    if k_batch != obs_batch:
        reps = k_batch // obs_batch
        # KVCache leaves are (layers, batch, seq, heads, head_dim): batch axis 1.
        kv_cache = jax.tree.map(lambda x: jnp.repeat(x, reps, axis=1), kv_cache)
        prefix_mask_k = jnp.repeat(prefix_mask, reps, axis=0)
        # pi0.5's embed_suffix does not read observation content (no state
        # token in the pi05 path), but tile defensively for shape consistency.
        observation_k = jax.tree.map(lambda x: jnp.repeat(x, reps, axis=0), observation)
    else:
        prefix_mask_k = prefix_mask
        observation_k = observation

    def euler_step(x_t, time):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation_k, x_t, jnp.broadcast_to(time, k_batch)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask_k, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        pos = jnp.sum(prefix_mask_k, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=pos,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return x_t + dt * v_t

    hdc_norm_bias = _hdc_bias(guidance, k_batch)  # (K, 3)

    def body(k, carry):
        x_t, time, _ = carry
        x_t = euler_step(x_t, time)
        w_hdc = jnp.where((k >= 2) & (k <= 5), 0.25, 0.0) * guidance.enabled
        x_t = x_t.at[..., :3].add(w_hdc * hdc_norm_bias[:, None, :])
        x_t = _repulsor(x_t, guidance)  # candidate C (eta=0 -> identity)
        x_t = _tilt_guidance(x_t, guidance)  # derived operator (lambda=0 -> identity)
        x_t, diag = _dcbf_repair(x_t, guidance, k)
        x_t = _fk_resample(x_t, guidance, k)  # candidate B (beta=0 -> identity)
        return x_t, time + dt, diag

    _, init_diag = _dcbf_repair(
        noise, guidance._replace(enabled=jnp.zeros_like(guidance.enabled)), num_steps - 1
    )
    x_0, _, last_diag = jax.lax.fori_loop(0, num_steps, body, (noise, 1.0, init_diag))

    if k_batch == 1:
        return x_0, last_diag

    # Lexicographic selection over candidates via scalarization: executed-prefix
    # feasibility dominates, then prefix clearance margin, then task progress.
    acc = _prefix_acceptance(x_0, guidance, prefix_len)
    score = (
        1e3 * acc["feasible"].astype(jnp.float32)
        + 10.0 * acc["min_margin"]
        + 0.1 * acc["disp_norm"]
        + guidance.progress_weight * acc["progress"]
        + guidance.lookahead_weight * acc["tail_margin"]
    )
    best = jnp.argmax(score)
    selected = jax.lax.dynamic_slice_in_dim(x_0, best, 1, axis=0)
    diag = {k: jax.lax.dynamic_slice_in_dim(v, best, 1, axis=0) for k, v in last_diag.items()}
    diag["feasible_count"] = jnp.broadcast_to(jnp.sum(acc["feasible"].astype(jnp.float32)), (1,))
    diag["selected_margin"] = jax.lax.dynamic_slice_in_dim(acc["min_margin"], best, 1, axis=0)
    # Epistemic telemetry: cross-candidate dispersion of the executed-prefix
    # translation (normalized action space) and margin spread across seeds.
    # Training-free OOD signals — logged only, never used for control here.
    prefix_xyz = x_0[:, :prefix_len, :3]
    diag["k_dispersion"] = jnp.broadcast_to(
        jnp.mean(jnp.linalg.norm(jnp.std(prefix_xyz, axis=0), axis=-1)), (1,)
    )
    diag["margin_spread"] = jnp.broadcast_to(
        jnp.max(acc["min_margin"]) - jnp.min(acc["min_margin"]), (1,)
    )
    # RMCS Phase 0 diagnostics: per-candidate executed-prefix displacement and
    # margin, plus the selected index. Logged only — never used for control.
    diag["cand_disp"] = acc["net_disp"]  # (K, 3)
    diag["cand_min_margin"] = acc["min_margin"]  # (K,)
    diag["cand_best_idx"] = jnp.broadcast_to(best.astype(jnp.float32), (1,))
    if return_candidates:
        return selected, diag, x_0, acc, last_diag
    return selected, diag


def adjoint_sample_actions(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    guidance: GuidanceParams,
    num_steps: int = 10,
    num_candidates: int = 8,
    prefix_len: int = 5,
    ascent_steps: int = 5,
    ascent_lr: float = 0.05,
    ascent_tau: float = 0.02,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
) -> tuple[_model.Actions, dict]:
    """Enforcement candidate A (stage 2): noise-space adjoint optimization.

    NO in-denoising repair — the policy flow is never overridden. Enforcement
    is a search over initial noises: K seeds are decoded unguided and scored
    by the SAME executed-prefix margin the certificate checks
    (`_prefix_acceptance`); the winner's noise is then refined by
    `ascent_steps` gradient-ascent steps of that margin w.r.t. the noise
    (reverse-mode through the 10-step Euler map; prefix KV constant).
    Ascent is GATED on margin < ascent_tau so already-safe chunks pass through
    bit-untouched — enforcement only engages when the certificate is at risk.
    Emitted actions are exactly F(z*): on the policy manifold by construction.
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    if noise is None:
        noise = jax.random.normal(
            rng, (num_candidates, self.action_horizon, self.action_dim))

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    kv_cache = jax.lax.stop_gradient(kv_cache)

    def denoise_one(z):
        """(H, ad) noise -> (1, H, ad) chunk; unrolled for clean reverse AD."""
        x_t, time_t = z[None], 1.0
        for _ in range(num_steps):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time_t, 1))
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            pos = (jnp.sum(prefix_mask, axis=-1)[:, None]
                   + jnp.cumsum(suffix_mask, axis=-1) - 1)
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=pos,
                kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            x_t = x_t + dt * v_t
            time_t = time_t + dt
        return x_t

    def margin_of(z):
        return _prefix_acceptance(denoise_one(z), guidance, prefix_len)["min_margin"][0]

    margins = jax.lax.map(margin_of, noise)  # sequential: memory-light K sweep
    best = jnp.argmax(margins)
    z0 = noise[best]

    grad_fn = jax.value_and_grad(margin_of)

    def ascend(_, z):
        m, g_z = grad_fn(z)
        gate = (m < ascent_tau).astype(jnp.float32) * guidance.enabled
        return z + ascent_lr * gate * g_z / (jnp.linalg.norm(g_z) + 1e-8)

    z_star = jax.lax.fori_loop(0, ascent_steps, ascend, z0)
    x_0 = denoise_one(z_star)
    acc = _prefix_acceptance(x_0, guidance, prefix_len)
    diag = {
        "selected_margin": acc["min_margin"],
        "feasible_count": jnp.broadcast_to(
            acc["feasible"].astype(jnp.float32)[0], (1,)),
        "seed_margin_best": jnp.broadcast_to(margins[best], (1,)),
        "ascent_gain": acc["min_margin"] - margins[best],
        "noise_shift": jnp.broadcast_to(jnp.linalg.norm(z_star - z0), (1,)),
    }
    return x_0, diag


def flow_decode_actions(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    guidance: GuidanceParams,
    num_steps: int = 10,
    num_candidates: int = 8,
    prefix_len: int = 5,
    start_time: float = 1.0,
    start_chunk: at.Float[at.Array, "ah ad"] | None = None,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
) -> tuple[_model.Actions, dict]:
    """Repair-free K-candidate decode with an optional SDEdit entry point.

    start_time=1.0: plain decode from fresh noise (the rejection-sampling /
    select-only path). start_time=t*<1 with start_chunk: initialize at
    x_{t*} = t*·ε + (1−t*)·start_chunk (the flow's own linear interpolant)
    and integrate only the remaining round(N·t*) Euler steps — the RENOISE
    operator: a local, on-manifold edit of a rejected chunk whose edit
    radius grows with t*. Returns all K chunks + their certified prefix
    margins; accept/reject and depth escalation live at the host level
    (guided_policy), keeping this function a pure jit graph per (K, t*).
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    n_run = max(1, int(round(num_steps * start_time)))
    dt = -1.0 / num_steps
    if noise is None:
        noise = jax.random.normal(
            rng, (num_candidates, self.action_horizon, self.action_dim))
    if start_chunk is None:
        x_t = noise
    else:
        x_t = start_time * noise + (1.0 - start_time) * start_chunk[None]

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    k_batch = x_t.shape[0]
    if k_batch != 1:
        kv_cache = jax.tree.map(lambda x: jnp.repeat(x, k_batch, axis=1), kv_cache)
        prefix_mask_k = jnp.repeat(prefix_mask, k_batch, axis=0)
        observation_k = jax.tree.map(lambda x: jnp.repeat(x, k_batch, axis=0), observation)
    else:
        prefix_mask_k, observation_k = prefix_mask, observation

    time_t = start_time
    for _ in range(n_run):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation_k, x_t, jnp.broadcast_to(time_t, k_batch))
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask_k, "b p -> b s p",
                                    s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        pos = (jnp.sum(prefix_mask_k, axis=-1)[:, None]
               + jnp.cumsum(suffix_mask, axis=-1) - 1)
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=pos,
            kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        x_t = x_t + dt * v_t
        time_t = time_t + dt

    acc = _prefix_acceptance(x_t, guidance, prefix_len)
    return x_t, {"min_margin": acc["min_margin"],
                 "feasible": acc["feasible"].astype(jnp.float32)}
