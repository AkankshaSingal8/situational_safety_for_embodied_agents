"""
flow_cbf_sample.py

Eager-mode reimplementation of Pi0.sample_actions()'s Euler denoising loop,
built to inject a per-step SLSQP barrier correction -- the mechanism from
English et al. (arXiv:2607.01378), "Neuro-Symbolic Safety Guidance for VLA
Models via Constrained Flow Matching".

WHY THIS FILE EXISTS (does not edit pi0.py): the original
Pi0.sample_actions() (vlsa-aegis/openpi/src/openpi/models/pi0.py:217-277)
runs its denoising loop via `jax.lax.while_loop`, a JAX-TRACED control-flow
primitive compiled to XLA. scipy's SLSQP (needed for the paper's
minimum-norm constrained correction) is not JAX-traceable -- it can't be
called from inside a traced while_loop body. This module reimplements the
SAME computation (embed_suffix -> attention -> PaliGemma.llm ->
action_out_proj -> Euler step) as a plain eager Python for-loop over
num_steps, so a correction can run between steps using the model's own
kv_cache / embed_prefix / embed_suffix / action_out_proj -- all called via
their public methods, unmodified.

CORRECTNESS: eager_sample_actions() must produce numerically identical
output to the traced sample_actions() when correction is disabled (same
rng/noise, same num_steps). validate_against_traced() below checks this
before any real eval is trusted -- run it first.

Paper's exact formulation (Appendix A's Algorithm 1, confirmed via full-text
fetch of arXiv:2607.01378, all 18 pages):
  - Discrete-time exponential CBF: B_j >= (1 - gamma) * B_{j-1}, gamma=0.9
  - Collision geometry: ellipsoid end-effector (semi-axes 0.06/0.12/0.11m)
    vs. sphere obstacle, signed-distance approx
  - Predicted eef trajectory from cumulative velocity integration over the
    FULL H-step action chunk, from the REAL current eef position:
    p_ee(0) = current position; p_ee(i) = p_ee(0) + dt * sum_{k=1}^{i} a_k[:3]
    for i=1..H -- H+1 points total, recomputed fresh at every denoising step
    (NOT carried over from the previous denoising iteration's corrected
    trajectory -- p_ee(0) is always the real robot state).
  - Correction: ONE joint min_delta ||delta||^2 s.t. ALL H constraints
    B_j >= (1-gamma)*B_{j-1} (j=1..H) simultaneously, via SLSQP, delta in
    R^{H x 3} (30 variables for H=10, matching the paper's own count).
  - Safety margin dsafe = 0.01m

FIXED (see git history for the earlier, incorrect version): an earlier
version of this file only corrected the FIRST horizon step's velocity per
denoising iteration (a 3-variable, 1-constraint problem) and carried the
predicted eef position across denoising iterations rather than
re-predicting the full trajectory fresh from the real current position each
time. That was structurally a repeated single-action filter, not the
paper's trajectory-level joint correction -- the entire point of their
method vs. AEGIS. correct_trajectory()/predict_trajectory() below now match
Algorithm 1 line-for-line: full (H,3) delta, all H constraints solved
jointly in one SLSQP call, p_ee(0) fixed at the real eef position for the
whole denoising loop.
"""

import dataclasses

import einops
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask

GAMMA = 0.9
DSAFE = 0.01
EEF_SEMI_AXES = np.array([0.06, 0.12, 0.11])


def _ellipsoid_sdf(p_ee, p_obs, r_obs, semi_axes=EEF_SEMI_AXES):
    """Signed distance approx: ellipsoid (eef) vs sphere (obstacle).
    Positive = safe (outside combined boundary), matches h>0=safe convention
    used throughout this project's CBF code (vlsa-aegis/main/utils.py's
    compute_h_ij uses the same sign convention)."""
    d = p_ee - p_obs
    scaled = d / semi_axes
    ellipsoid_dist = np.linalg.norm(scaled) - 1.0  # >0 outside unit ellipsoid in scaled space
    # Convert scaled-space distance back to an approximate world-space margin,
    # then subtract obstacle radius + safety margin.
    world_scale = np.linalg.norm(d) / (np.linalg.norm(scaled) + 1e-9) if np.linalg.norm(scaled) > 1e-9 else 1.0
    return ellipsoid_dist * world_scale - r_obs - DSAFE


def predict_trajectory(eef_pos0, v_chunk_xyz, dt_control):
    """Eq 7: p_ee^(0) = current real eef position; p_ee^(i) = p_ee(0) + dt *
    cumsum_{k=1}^{i} a_k[:3] for i=1..H. Returns array of shape (H+1, 3):
    the FULL predicted trajectory, not just the next step."""
    cumsum = np.cumsum(v_chunk_xyz, axis=0)
    H = v_chunk_xyz.shape[0]
    traj = np.zeros((H + 1, 3))
    traj[0] = eef_pos0
    traj[1:] = eef_pos0 + dt_control * cumsum
    return traj


def barrier_values(traj, p_obs, r_obs):
    """B_j for j=0..H (Eq 8), one per predicted trajectory point."""
    return np.array([_ellipsoid_sdf(p, p_obs, r_obs) for p in traj])


def correct_trajectory(v_chunk_xyz, eef_pos0, dt_control, p_obs, r_obs):
    """Algorithm 1 / Eq 9-10: joint minimum-norm correction delta in R^{H x 3}
    over the WHOLE predicted trajectory, solved in ONE SLSQP call subject to
    all H consecutive-point barrier constraints simultaneously -- not H
    independent single-step corrections. eef_pos0 is the REAL current
    end-effector position (constant for this denoising step; NOT carried
    over from a previous denoising iteration's corrected trajectory)."""
    H = v_chunk_xyz.shape[0]

    def barriers_of(delta_flat):
        delta = delta_flat.reshape(H, 3)
        traj = predict_trajectory(eef_pos0, v_chunk_xyz + delta, dt_control)
        return barrier_values(traj, p_obs, r_obs)  # length H+1

    b0 = barriers_of(np.zeros(H * 3))
    violated = any(b0[j] < (1.0 - GAMMA) * b0[j - 1] for j in range(1, H + 1))
    if not violated:
        return v_chunk_xyz, b0

    def objective(delta_flat):
        return float(np.dot(delta_flat, delta_flat))

    def make_constraint(j):
        # B_j >= (1-gamma)*B_{j-1}, j=1..H (Eq 9), each a separate SLSQP
        # inequality constraint evaluated against the SAME delta vector.
        def c(delta_flat):
            b = barriers_of(delta_flat)
            return b[j] - (1.0 - GAMMA) * b[j - 1]

        return c

    constraints = [{"type": "ineq", "fun": make_constraint(j)} for j in range(1, H + 1)]
    res = minimize(
        objective,
        x0=np.zeros(H * 3),
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 50, "ftol": 1e-8},
    )
    delta = (res.x if res.success else np.zeros(H * 3)).reshape(H, 3)
    corrected = v_chunk_xyz + delta
    b_final = barriers_of(delta.flatten())
    return corrected, b_final


def eager_sample_actions(
    model,
    rng,
    observation,
    eef_pos_world,
    p_obs=None,
    r_obs=0.06,
    dt_control=0.05,
    num_steps=10,
    noise=None,
    apply_correction=True,
):
    """Reimplements Pi0.sample_actions()'s Euler loop eagerly. When
    p_obs is None, behaves identically to the traced version (no
    correction possible -- used for validate_against_traced()).
    eef_pos_world: current world-frame end-effector position (np.ndarray,
    shape (3,)), used to seed the predicted-trajectory integration."""
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    x_t = noise
    time = 1.0
    # Real current end-effector position, held CONSTANT across all N
    # denoising iterations -- Algorithm 1 re-predicts the full trajectory
    # from this same real p_ee^(0) at every step; it is not advanced by the
    # previous iteration's correction (that would conflate the denoising-step
    # loop with the trajectory-point loop, which are different in the paper).
    eef_pos0 = np.array(eef_pos_world, dtype=np.float64)

    for _ in range(num_steps):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_s = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask_s, suffix_attn_mask], axis=-1)
        step_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=step_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])

        if apply_correction and p_obs is not None:
            v_xyz = np.asarray(v_t[0, :, :3], dtype=np.float64)  # first batch elem, ALL H steps' translational velocity
            corrected_xyz, _ = correct_trajectory(v_xyz, eef_pos0, dt_control, p_obs, r_obs)
            v_t = v_t.at[0, :, :3].set(jnp.asarray(corrected_xyz))

        x_t = x_t + dt * v_t
        time = time + dt

    return x_t


def validate_against_traced(model, rng, observation, num_steps=10):
    """Sanity check: with correction disabled, eager loop must match
    Pi0.sample_actions()'s traced jax.lax.while_loop output within float
    tolerance. Run this before trusting any real eval output."""
    noise = jax.random.normal(rng, (observation.state.shape[0], model.action_horizon, model.action_dim))
    traced_out = model.sample_actions(rng, observation, num_steps=num_steps, noise=noise)
    eager_out = eager_sample_actions(
        model, rng, observation, eef_pos_world=np.zeros(3), p_obs=None, num_steps=num_steps, noise=noise
    )
    diff = float(jnp.max(jnp.abs(traced_out - eager_out)))
    return diff, traced_out, eager_out
