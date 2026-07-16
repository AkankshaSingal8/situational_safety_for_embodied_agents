"""ROCS v1 — Runtime Outcome-Conditioned Steering (release leg).

Steers OUTCOMES, not just contacts: parses the action chunk for a predicted
release event (gripper opening) and, if the predicted landing point falls in a
forbidden region (FORBIDDEN_ON / RELEASE_OVER relations from the semantic
layer), delays the release until the trajectory exits the region.

Gripper convention (verified 2026-07-16 against the SafeLIBERO client, which
passes server actions straight to robosuite `env.step`): action dim 6 in
[-1, 1], **+1 = close, -1 = open**. A release event is the first step whose
gripper command crosses from > 0 to <= 0.

All functions are pure jnp and operate on the DENORMALIZED command chunk
(B, H, 7) — translation dims 0:3 in [-1,1] commands, gripper dim 6. They
compose with the DCBF repair (which owns dims 0:3); ROCS only touches dim 6.
"""

import jax.numpy as jnp

GRIPPER_DIM = 6


def release_step(chunk: jnp.ndarray) -> jnp.ndarray:
    """First step index where the gripper command crosses to open (<= 0)
    from a closed (> 0) previous command; H if the chunk never releases.

    chunk: (B, H, >=7) command-space actions. Returns (B,) int32.
    """
    g = chunk[..., GRIPPER_DIM]  # (B, H)
    prev = jnp.concatenate([jnp.ones_like(g[:, :1]), g[:, :-1]], axis=1)
    release = (prev > 0.0) & (g <= 0.0)  # (B, H)
    h = g.shape[1]
    idx = jnp.argmax(release, axis=1)
    any_rel = jnp.any(release, axis=1)
    return jnp.where(any_rel, idx, h).astype(jnp.int32)


def landing_positions(chunk: jnp.ndarray, eef_pos: jnp.ndarray,
                      translation_scale: float) -> jnp.ndarray:
    """EEF rollout position at each step (single-integrator, same model as the
    DCBF repair). chunk (B, H, >=7), eef_pos (3,) -> (B, H, 3)."""
    disp = chunk[..., :3] * translation_scale
    return eef_pos + jnp.cumsum(disp, axis=1)


def forbidden_release(chunk: jnp.ndarray, eef_pos: jnp.ndarray,
                      translation_scale: float, center: jnp.ndarray,
                      radius: jnp.ndarray) -> jnp.ndarray:
    """(B,) bool: chunk releases AND the release point is horizontally inside
    the forbidden disc (xy distance < radius) — 'over' semantics: dropping
    from any height above the region counts."""
    h = chunk.shape[1]
    rs = release_step(chunk)  # (B,)
    p = landing_positions(chunk, eef_pos, translation_scale)  # (B, H, 3)
    idx = jnp.clip(rs, 0, h - 1)
    p_rel = jnp.take_along_axis(p, idx[:, None, None].repeat(3, -1), axis=1)[:, 0]
    inside = jnp.linalg.norm((p_rel - center)[..., :2], axis=-1) < radius
    return (rs < h) & inside


def delay_release(chunk: jnp.ndarray, eef_pos: jnp.ndarray,
                  translation_scale: float, center: jnp.ndarray,
                  radius: jnp.ndarray, enabled: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    """Release-delay clamp: on chunks whose predicted release lands inside the
    forbidden disc, hold the gripper CLOSED (+1) at every step whose rollout
    position is still inside the disc. The release then happens at the first
    outside step the policy itself opens on (or is pushed to the next chunk —
    receding horizon replans anyway). Translation dims are untouched, so DCBF
    certification of dims 0:3 is preserved.

    Returns (modified chunk, diagnostics).
    """
    bad = forbidden_release(chunk, eef_pos, translation_scale, center, radius)  # (B,)
    p = landing_positions(chunk, eef_pos, translation_scale)
    inside = jnp.linalg.norm((p - center)[..., :2], axis=-1) < radius  # (B, H)
    hold = bad[:, None] & inside  # (B, H)
    g = chunk[..., GRIPPER_DIM]
    g_new = jnp.where(hold & (g <= 0.0), 1.0, g)
    out = chunk.at[..., GRIPPER_DIM].set(
        enabled * g_new + (1.0 - enabled) * g
    )
    diag = {
        "rocs_forbidden_release": bad.astype(jnp.float32),
        "rocs_held_steps": jnp.sum((hold & (g <= 0.0)).astype(jnp.float32), axis=1),
    }
    return out, diag
