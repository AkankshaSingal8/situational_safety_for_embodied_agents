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


def _dcbf_repair(x_t: jnp.ndarray, g: GuidanceParams) -> tuple[jnp.ndarray, dict]:
    """Repair the (B, H, action_dim) normalized chunk to satisfy the DCBF chain.

    Only dims 0:3 (translational commands) are modified. Returns the repaired
    chunk and diagnostics. The correction is scaled by `g.enabled` so a single
    jitted code path serves both guided and unguided requests.
    """
    b, h, _ = x_t.shape

    # normalized -> env command in [-1, 1] (quantile unnormalize, cf.
    # transforms.Unnormalize._unnormalize_quantile)
    span = g.q99 - g.q01 + 1e-6
    cmd = (x_t[..., :3] + 1.0) / 2.0 * span + g.q01  # (B, H, 3)
    disp = cmd * g.translation_scale  # metric per-step displacement

    # Vertical companion points approximating the full manipulated system, not
    # just the EEF frame origin: the hand/wrist body sits above the grip site
    # (v19 forensics: unmonitored hand plowed obstacles), fingers and a carried
    # object hang below. Each shares r_eff; the barrier is the min over points.
    COMPANIONS = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.08], [0.0, 0.0, -0.06]])

    def _closest(p):
        """Min distance (and its direction) from any companion point to the obstacle."""
        diffs = p[None, :] + COMPANIONS - g.obstacle_pos  # (3, 3)
        dists = jnp.linalg.norm(diffs, axis=-1)
        k = jnp.argmin(dists)
        return dists[k], diffs[k] / (dists[k] + 1e-8)

    def repair_one(disp_b):
        """Sweep j = 0..H-1 enforcing B_j >= (1-gamma) * B_{j-1}."""
        p0 = g.eef_pos
        d0, _ = _closest(p0)
        b0 = d0 - g.r_eff

        def body(j, carry):
            disp_acc, p_prev, b_prev = carry
            r_j = g.r_eff + g.inflation_slope * (j + 1.0)
            p_j = p_prev + disp_acc[j]
            dist, direction = _closest(p_j)
            b_j = dist - r_j
            need = (1.0 - g.gamma) * b_prev
            deficit = jnp.maximum(need - b_j, 0.0)
            # Push p_j (and, through the cumulative rollout, all later points)
            # away from the obstacle along the closest companion's direction.
            new_step = disp_acc[j] + deficit * direction
            # Respect the actuator range: |command| <= 1 per dim.
            new_cmd = jnp.clip(new_step / g.translation_scale, -1.0, 1.0)
            new_step = new_cmd * g.translation_scale
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
    return x_new, diag


def guided_sample_actions(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    guidance: GuidanceParams,
    num_steps: int = 10,
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
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def euler_step(x_t, time):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=pos,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return x_t + dt * v_t

    def body(_, carry):
        x_t, time, _ = carry
        x_t = euler_step(x_t, time)
        x_t, diag = _dcbf_repair(x_t, guidance)
        return x_t, time + dt, diag

    _, init_diag = _dcbf_repair(noise, guidance._replace(enabled=jnp.zeros_like(guidance.enabled)))
    x_0, _, last_diag = jax.lax.fori_loop(0, num_steps, body, (noise, 1.0, init_diag))
    return x_0, last_diag
