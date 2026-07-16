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

    def _closest(p):
        """Min distance (and its direction) from any companion point to the obstacle."""
        diffs = p[None, :] + COMPANIONS - g.obstacle_pos  # (3, 3)
        dists = jnp.linalg.norm(diffs, axis=-1)
        k = jnp.argmin(dists)
        return dists[k], diffs[k] / (dists[k] + 1e-8)

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
            r_j = ms_k * (g.r_eff + g.inflation_slope * (j + 1.0)) * _corridor_scale(p_j, g)
            dist, direction = _closest(p_j)
            b_j = dist - r_j
            need = (1.0 - g.gamma) * b_prev
            deficit = jnp.maximum(need - b_j, 0.0)
            # Push p_j (and, through the cumulative rollout, all later points)
            # away from the obstacle along the closest companion's direction.
            pushed = orig_step + deficit * direction
            pushed_cmd = jnp.clip(pushed / g.translation_scale, -1.0, 1.0)
            pushed = pushed_cmd * g.translation_scale
            p_push = p_prev + pushed
            dist_push, _ = _closest(p_push)
            # FULL-BRAKE alternative: remove the toward-obstacle component of
            # the original step. Its radial component is non-negative, so the
            # barrier cannot decrease at this step — a clamp-independent floor.
            approach = jnp.maximum(jnp.dot(orig_step, -direction), 0.0)
            braked = orig_step + approach * direction
            braked_cmd = jnp.clip(braked / g.translation_scale, -1.0, 1.0)
            braked = braked_cmd * g.translation_scale
            dist_brake, _ = _closest(p_prev + braked)
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
            new_step_restituted_cmd = jnp.clip(new_step_restituted / g.translation_scale, -1.0, 1.0)
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
            dist2, _ = _closest(p_j)
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
    comp_dists = jnp.linalg.norm(
        p_traj[:, :, None, :] + COMPANIONS[None, None, :, :] - g.obstacle_pos, axis=-1
    )  # (B, H, 3)
    clearance = jnp.min(comp_dists, axis=-1) - r_horizon
    b0_val = jnp.min(jnp.linalg.norm(g.eef_pos[None, :] + COMPANIONS - g.obstacle_pos, axis=-1)) - g.r_eff
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
    dists = jnp.min(
        jnp.linalg.norm(p_traj[:, :, None, :] + companions[None, None] - g.obstacle_pos, axis=-1), axis=-1
    )  # (K, H)
    h = disp.shape[1]
    r_horizon = (g.r_eff + g.inflation_slope * jnp.arange(1.0, h + 1.0)) * _corridor_scale(p_traj, g)
    b = dists - r_horizon  # (K, H)
    b0 = jnp.min(jnp.linalg.norm(g.eef_pos[None, :] + companions - g.obstacle_pos, axis=-1)) - g.r_eff
    chain = jnp.concatenate([jnp.broadcast_to(b0, (b.shape[0], 1)), b[:, :prefix_len]], axis=1)
    viol = (1.0 - g.gamma) * chain[:, :-1] - chain[:, 1:]
    feasible = jnp.max(viol, axis=1) <= 1e-4  # (K,)
    min_margin = jnp.min(b[:, :prefix_len], axis=1)
    disp_norm = jnp.linalg.norm(jnp.sum(disp[:, :prefix_len], axis=1), axis=-1)
    return {"feasible": feasible, "min_margin": min_margin, "disp_norm": disp_norm}


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
) -> tuple[_model.Actions, dict]:
    """`Pi0.sample_actions` with a DCBF repair after every Euler step.

    Bind to a loaded Pi0 instance via `types.MethodType(guided_sample_actions,
    model)`. The denoising math replicates upstream `sample_actions` exactly
    (KV-cache prefix, suffix attention, Euler update); the only addition is the
    `_dcbf_repair` call inside the loop, so subsequent denoising steps adapt to
    the correction — the property that distinguishes in-generation guidance
    from post-hoc filtering.
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

    def body(k, carry):
        x_t, time, _ = carry
        x_t = euler_step(x_t, time)
        x_t, diag = _dcbf_repair(x_t, guidance, k)
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
    score = 1e3 * acc["feasible"].astype(jnp.float32) + 10.0 * acc["min_margin"] + 0.1 * acc["disp_norm"]
    best = jnp.argmax(score)
    selected = jax.lax.dynamic_slice_in_dim(x_0, best, 1, axis=0)
    diag = {k: jax.lax.dynamic_slice_in_dim(v, best, 1, axis=0) for k, v in last_diag.items()}
    diag["feasible_count"] = jnp.broadcast_to(jnp.sum(acc["feasible"].astype(jnp.float32)), (1,))
    diag["selected_margin"] = jax.lax.dynamic_slice_in_dim(acc["min_margin"], best, 1, axis=0)
    return selected, diag
