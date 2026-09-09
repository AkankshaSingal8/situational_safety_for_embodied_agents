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

WHERE THE CORRECTION LIVES NOW: not in this file's denoising loop. Applying
the barrier to v_t was category-wrong -- v_t is the velocity FIELD
(v_t ~ eps - x_1 in a pi0-style flow model), not an action, and the
Unnormalize output transform has not yet run, so metre-valued obstacle
geometry cannot be compared against it. The barrier now acts on the DECODED,
UNNORMALISED action chunk: see flow_cbf_barrier.correct_action_chunk and its
call site in run_libsafety_eval_flowcbf._infer_with_flow_cbf. Passing
apply_correction=True here now raises rather than silently returning an
unfiltered chunk.

Barrier math (Algorithm 1: full (H,3) delta, all H constraints solved jointly
in one SLSQP call, p_ee(0) fixed at the real eef position) lives in
flow_cbf_barrier.py, which is jax-free and unit-tested at the true 20 Hz
control dt.
"""

import dataclasses
import logging

import einops
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask

logger = logging.getLogger(__name__)

# Barrier math lives in flow_cbf_barrier.py (jax-free, unit-tested). Re-exported
# here so existing importers keep working.
from flow_cbf_barrier import (  # noqa: E402
    DSAFE,
    DYNAMICS_GAIN,
    EEF_SEMI_AXES,
    GAMMA,
    assert_barrier_is_active,
    barrier_values,
    correct_action_chunk,
    ellipsoid_sdf as _ellipsoid_sdf,
    predict_trajectory,
)


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
    if apply_correction:
        # Fail loudly rather than silently returning an unfiltered chunk: this
        # function used to correct v_t in-loop, which was category-wrong. The
        # caller must now correct the decoded chunk instead.
        raise ValueError(
            "eager_sample_actions no longer applies the barrier: v_t is a "
            "velocity field, not an action, and Unnormalize has not run. Call "
            "flow_cbf_barrier.correct_action_chunk on the decoded, unnormalised "
            "chunk instead, and pass apply_correction=False here."
        )
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

        # NO correction here. v_t is the velocity FIELD (v_t ~ eps - x_1 in a
        # pi0-style flow model), not an action, and Unnormalize has not run, so
        # a metre-valued barrier cannot be applied to it. The correction now
        # runs on the decoded, unnormalised chunk -- see
        # flow_cbf_barrier.correct_action_chunk and its call site in
        # run_libsafety_eval_flowcbf._infer_with_flow_cbf.
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
