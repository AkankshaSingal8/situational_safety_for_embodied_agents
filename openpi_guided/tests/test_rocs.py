import jax.numpy as jnp
import numpy as np

from openpi.models.rocs import (
    delay_release,
    forbidden_release,
    landing_positions,
    release_step,
)

SCALE = 0.05


def chunk_with_release(rel_at, h=10, vel=(1.0, 0.0, 0.0)):
    """Move at `vel` command, gripper closed (+1) until rel_at then open (-1)."""
    c = np.zeros((1, h, 7), dtype=np.float32)
    c[..., :3] = np.asarray(vel)
    c[..., 6] = 1.0
    if rel_at is not None:
        c[0, rel_at:, 6] = -1.0
    return jnp.asarray(c)


def test_release_step_detection():
    assert int(release_step(chunk_with_release(4))[0]) == 4
    assert int(release_step(chunk_with_release(0))[0]) == 0
    assert int(release_step(chunk_with_release(None))[0]) == 10  # never releases


def test_landing_positions_rollout():
    p = landing_positions(chunk_with_release(None), jnp.zeros(3), SCALE)
    np.testing.assert_allclose(np.asarray(p[0, -1]), [0.5, 0.0, 0.0], atol=1e-6)


def test_forbidden_release_inside_and_outside():
    eef = jnp.zeros(3)
    center = jnp.asarray([0.25, 0.0, 0.0])  # release at step 4 -> x=0.25
    r = jnp.float32(0.10)
    assert bool(forbidden_release(chunk_with_release(4), eef, SCALE, center, r)[0])
    far = jnp.asarray([5.0, 5.0, 0.0])
    assert not bool(forbidden_release(chunk_with_release(4), eef, SCALE, far, r)[0])
    # 'over' semantics: same xy, big z offset still forbidden
    high = jnp.asarray([0.25, 0.0, 3.0])
    assert bool(forbidden_release(chunk_with_release(4), eef, SCALE, high, r)[0])


def test_delay_release_holds_until_outside():
    eef = jnp.zeros(3)
    center = jnp.asarray([0.25, 0.0, 0.0])
    r = jnp.float32(0.08)  # inside for x in (0.17, 0.33) -> steps 3..5 (x=.20,.25,.30)
    out, diag = delay_release(chunk_with_release(4), eef, SCALE, center, r,
                              jnp.float32(1.0))
    g = np.asarray(out[0, :, 6])
    assert (g[:6] == 1.0).all()  # held closed through the disc (steps 4,5 flipped)
    assert (g[6:] == -1.0).all()  # releases once outside
    assert float(diag["rocs_forbidden_release"][0]) == 1.0
    assert float(diag["rocs_held_steps"][0]) == 2.0
    # translation untouched
    np.testing.assert_array_equal(np.asarray(out[..., :3]),
                                  np.asarray(chunk_with_release(4)[..., :3]))


def test_delay_release_noop_when_safe_or_disabled():
    eef = jnp.zeros(3)
    far = jnp.asarray([5.0, 5.0, 0.0])
    ch = chunk_with_release(4)
    out, _ = delay_release(ch, eef, SCALE, far, jnp.float32(0.10), jnp.float32(1.0))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ch))
    center = jnp.asarray([0.25, 0.0, 0.0])
    out, _ = delay_release(ch, eef, SCALE, center, jnp.float32(0.10), jnp.float32(0.0))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ch))
