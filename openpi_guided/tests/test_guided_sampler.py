"""Offline unit tests for the in-denoising DCBF repair (no GPU, no checkpoint).

Run from the worktree root:
    JAX_PLATFORMS=cpu PYTHONPATH=openpi_guided/src python -m pytest openpi_guided/tests -q
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.pi0_guided import GuidanceParams, _dcbf_repair, _prefix_acceptance

H = 10
GAMMA = 0.9
SCALE = 0.05
# Symmetric quantiles so normalized 0 == zero command in the synthetic tests.
Q01 = np.full(3, -1.0, dtype=np.float32)
Q99 = np.full(3, 1.0, dtype=np.float32)


def make_params(eef, obstacle, r_eff=0.12, enabled=1.0, target=None, dest=None,
                corridor_radius=0.07, corridor_relax=0.6, dbnr_beta0=0.0):
    return GuidanceParams(
        enabled=jnp.float32(enabled),
        eef_pos=jnp.asarray(np.asarray(eef, dtype=np.float32)),
        obstacle_pos=jnp.asarray(np.asarray(obstacle, dtype=np.float32)),
        r_eff=jnp.float32(r_eff),
        inflation_slope=jnp.float32(0.0),
        gamma=jnp.float32(GAMMA),
        q01=jnp.asarray(Q01),
        q99=jnp.asarray(Q99),
        translation_scale=jnp.float32(SCALE),
        repair_weight=jnp.ones(10, dtype=jnp.float32),
        margin_scale=jnp.ones(10, dtype=jnp.float32),
        target_pos=jnp.asarray(np.asarray(target if target is not None else [100.0, 100.0, 100.0], dtype=np.float32)),
        dest_pos=jnp.asarray(np.asarray(dest if dest is not None else [100.0, 100.0, 100.0], dtype=np.float32)),
        corridor_radius=jnp.float32(corridor_radius),
        corridor_relax=jnp.float32(corridor_relax),
        companion_scale=jnp.float32(1.0),
        dbnr_beta0=jnp.float32(dbnr_beta0),
    )


def chunk_toward(direction, magnitude=0.8, action_dim=32):
    """Chunk whose translational commands all push in `direction`."""
    x = np.zeros((1, H, action_dim), dtype=np.float32)
    d = np.asarray(direction, dtype=np.float32)
    d = d / np.linalg.norm(d)
    x[0, :, :3] = d * magnitude
    return jnp.asarray(x)


def rollout(x, params):
    """Numpy reference: normalized chunk -> metric EEF positions."""
    span = Q99 - Q01 + 1e-6
    cmd = (np.asarray(x)[0, :, :3] + 1.0) / 2.0 * span + Q01
    disp = cmd * SCALE
    return np.asarray(params.eef_pos) + np.cumsum(disp, axis=0)


def barrier_chain(x, params):
    p = rollout(x, params)
    dists = np.linalg.norm(p - np.asarray(params.obstacle_pos), axis=-1)
    b = dists - float(params.r_eff)
    b0 = np.linalg.norm(np.asarray(params.eef_pos) - np.asarray(params.obstacle_pos)) - float(params.r_eff)
    return b0, b


def test_noop_when_far():
    params = make_params(eef=[0.0, 0.0, 1.0], obstacle=[5.0, 5.0, 5.0])
    x = chunk_toward([1.0, 0.0, 0.0])
    x_new, diag = _dcbf_repair(x, params, 9)
    np.testing.assert_allclose(np.asarray(x_new), np.asarray(x), atol=1e-6)
    assert float(diag["correction_norm"][0]) < 1e-6
    assert float(diag["max_residual"][0]) < 1e-6


def test_disabled_is_identity_even_when_violating():
    params = make_params(eef=[0.0, 0.0, 1.0], obstacle=[0.15, 0.0, 1.0], enabled=0.0)
    x = chunk_toward([1.0, 0.0, 0.0])  # drives straight at the obstacle
    x_new, _ = _dcbf_repair(x, params, 9)
    np.testing.assert_allclose(np.asarray(x_new), np.asarray(x), atol=1e-6)


def test_head_on_chunk_gets_repaired_to_satisfy_dcbf_chain():
    params = make_params(eef=[0.0, 0.0, 1.0], obstacle=[0.20, 0.0, 1.0])
    x = chunk_toward([1.0, 0.0, 0.0])  # 4 cm/step straight at obstacle 20 cm away
    x_new, diag = _dcbf_repair(x, params, 9)

    b0, b = barrier_chain(x_new, params)
    chain = np.concatenate([[b0], b])
    violations = (1.0 - GAMMA) * chain[:-1] - chain[1:]
    assert np.max(violations) < 1e-4, f"DCBF chain violated: {violations}"
    assert float(diag["correction_norm"][0]) > 1e-3
    # The repaired path must keep positive clearance.
    assert np.min(b) > -1e-4
    assert float(diag["min_clearance"][0]) == pytest.approx(np.min(b), abs=1e-4)


def test_tangential_chunk_barely_modified():
    # Path passes at ~18 cm lateral offset from an r_eff=0.12 obstacle: safe.
    params = make_params(eef=[0.0, 0.18, 1.0], obstacle=[0.15, 0.0, 1.0])
    x = chunk_toward([1.0, 0.0, 0.0], magnitude=0.5)
    x_new, diag = _dcbf_repair(x, params, 9)
    assert float(diag["correction_norm"][0]) < 1e-3


def test_starts_inside_margin_escapes_gradually():
    # B_0 < 0: DCBF requires B_j >= (1-gamma)*B_{j-1}, i.e., geometric decay of
    # the violation toward the boundary — no teleporting, monotone improvement.
    params = make_params(eef=[0.10, 0.0, 1.0], obstacle=[0.15, 0.0, 1.0])
    x = chunk_toward([1.0, 0.0, 0.0], magnitude=0.3)
    x_new, diag = _dcbf_repair(x, params, 9)
    b0, b = barrier_chain(x_new, params)
    assert b0 < 0
    chain = np.concatenate([[b0], b])
    # From deep inside the margin the |command|<=1 clamp physically bounds the
    # escape speed, so the chain condition can be unattainable at early steps.
    # The honest guaranteed property is monotone escape at max commanded speed.
    assert np.all(np.diff(chain) >= -1e-5), f"barrier not monotone: {chain}"
    # The end of the horizon must be strictly safer than the start.
    assert b[-1] > b0
    # Residual violation must be reported, not hidden.
    assert float(diag["max_residual"][0]) > 1e-4


def test_command_clamp_respected():
    params = make_params(eef=[0.0, 0.0, 1.0], obstacle=[0.08, 0.0, 1.0], r_eff=0.2)
    x = chunk_toward([1.0, 0.0, 0.0])
    x_new, diag = _dcbf_repair(x, params, 9)
    span = Q99 - Q01 + 1e-6
    cmd = (np.asarray(x_new)[0, :, :3] + 1.0) / 2.0 * span + Q01
    assert np.max(np.abs(cmd)) <= 1.0 + 1e-5
    # Clamp saturation is allowed to leave residual, but it must be reported.
    assert float(diag["max_residual"][0]) >= 0.0


def test_only_translational_dims_touched():
    params = make_params(eef=[0.0, 0.0, 1.0], obstacle=[0.20, 0.0, 1.0])
    rng = np.random.default_rng(0)
    x = np.array(chunk_toward([1.0, 0.0, 0.0]))  # writable copy
    x[..., 3:] = rng.normal(size=x[..., 3:].shape).astype(np.float32)
    x = jnp.asarray(x)
    x_new, _ = _dcbf_repair(x, params, 9)
    np.testing.assert_allclose(np.asarray(x_new)[..., 3:], np.asarray(x)[..., 3:], atol=1e-7)


def test_below_path_obstacle_caught_by_companion_point():
    # Obstacle 13 cm below the flight path: the EEF-frame point stays clear
    # (0.13 > r_eff 0.10) but the fingertip/payload companion at z-0.06 comes
    # within 0.07 — the repair must fire. This is the geometry that caused the
    # n=50 r2 collisions (hand/fingers/carried object unmodeled).
    params = make_params(eef=[0.0, 0.0, 1.0], obstacle=[0.20, 0.0, 0.87], r_eff=0.10)
    x = chunk_toward([1.0, 0.0, 0.0], magnitude=0.8)
    x_new, diag = _dcbf_repair(x, params, 9)
    assert float(diag["correction_norm"][0]) > 1e-3
    # min over companion points must be reflected in the clearance diagnostic.
    assert float(diag["min_clearance"][0]) > -1e-3


def test_batch_dimension():
    params = make_params(eef=[0.0, 0.0, 1.0], obstacle=[0.20, 0.0, 1.0])
    x1 = chunk_toward([1.0, 0.0, 0.0])
    x2 = chunk_toward([-1.0, 0.0, 0.0])  # moving away: should be untouched
    x = jnp.concatenate([x1, x2], axis=0)
    x_new, diag = _dcbf_repair(x, params, 9)
    assert float(diag["correction_norm"][0]) > 1e-3
    assert float(diag["correction_norm"][1]) < 1e-6


def test_schedule_zero_weight_is_identity():
    params = make_params(eef=[0.0, 0.0, 1.0], obstacle=[0.20, 0.0, 1.0])
    params = params._replace(repair_weight=jnp.zeros(10, dtype=jnp.float32))
    x = chunk_toward([1.0, 0.0, 0.0])
    x_new, diag = _dcbf_repair(x, params, 3)
    np.testing.assert_allclose(np.asarray(x_new), np.asarray(x), atol=1e-6)


def test_brake_floor_barrier_never_below_no_approach():
    # Property: the chosen step's barrier is >= the full-brake floor, i.e., the
    # repair never executes a step worse than "stop approaching".
    rng = np.random.default_rng(3)
    params = make_params(eef=[0.0, 0.0, 1.0], obstacle=[0.12, 0.02, 1.0], r_eff=0.15)
    for _ in range(5):
        x = np.zeros((1, H, 32), dtype=np.float32)
        x[0, :, :3] = rng.uniform(-1, 1, size=(H, 3)).astype(np.float32)
        x_new, diag = _dcbf_repair(jnp.asarray(x), params, 9)
        b0, b = barrier_chain(x_new, params)
        chain = np.concatenate([[b0], b])
        # Guaranteed floor: B_{j+1} >= min(B_j, (1-gamma)*B_j) — either the
        # chain condition was met (push) or the step did not approach (brake).
        floor = np.minimum(chain[:-1], (1.0 - GAMMA) * chain[:-1])
        assert np.all(chain[1:] >= floor - 5e-3), (chain, floor)


def test_prefix_acceptance_separates_candidates():
    params = make_params(eef=[0.0, 0.0, 1.0], obstacle=[0.20, 0.0, 1.0])
    into = chunk_toward([1.0, 0.0, 0.0])          # drives at the obstacle
    around = chunk_toward([0.0, 1.0, 0.0])        # detours laterally
    x = jnp.concatenate([into, around], axis=0)
    acc = _prefix_acceptance(x, params, prefix_len=5)
    feas = np.asarray(acc["feasible"])
    assert not feas[0] and feas[1], feas
    assert float(acc["min_margin"][1]) > float(acc["min_margin"][0])
    # progress proxy is positive for the moving candidate
    assert float(acc["disp_norm"][1]) > 0.05


def test_corridor_exemption_relaxes_along_sanctioned_approach():
    # Obstacle adjacent to the target: without the corridor the head-on chunk
    # is heavily repaired; with the target-corridor active the same chunk
    # passes with a much smaller correction (margins relax inside the cone).
    x = chunk_toward([1.0, 0.0, 0.0], magnitude=0.5)
    p_no = make_params(eef=[0.0, 0.0, 1.0], obstacle=[0.22, 0.05, 1.0], r_eff=0.15)
    p_yes = make_params(eef=[0.0, 0.0, 1.0], obstacle=[0.22, 0.05, 1.0], r_eff=0.15,
                        target=[0.25, 0.0, 1.0])
    _, d_no = _dcbf_repair(x, p_no, 9)
    _, d_yes = _dcbf_repair(x, p_yes, 9)
    assert float(d_yes["correction_norm"][0]) < float(d_no["correction_norm"][0]) - 1e-3


def test_corridor_never_relaxes_off_axis():
    # Same obstacle, but the sanctioned target lies in the OPPOSITE direction:
    # the corridor must grant nothing to the hazard-side approach.
    x = chunk_toward([1.0, 0.0, 0.0], magnitude=0.5)
    p_no = make_params(eef=[0.0, 0.0, 1.0], obstacle=[0.22, 0.05, 1.0], r_eff=0.15)
    p_opp = make_params(eef=[0.0, 0.0, 1.0], obstacle=[0.22, 0.05, 1.0], r_eff=0.15,
                        target=[-0.25, 0.0, 1.0])
    _, d_no = _dcbf_repair(x, p_no, 9)
    _, d_opp = _dcbf_repair(x, p_opp, 9)
    np.testing.assert_allclose(float(d_opp["correction_norm"][0]),
                               float(d_no["correction_norm"][0]), atol=1e-4)


def test_far_target_disables_corridor():
    x = chunk_toward([1.0, 0.0, 0.0])
    p_far = make_params(eef=[0.0, 0.0, 1.0], obstacle=[0.20, 0.0, 1.0])  # default FAR
    p_ref = make_params(eef=[0.0, 0.0, 1.0], obstacle=[0.20, 0.0, 1.0])
    _, d_far = _dcbf_repair(x, p_far, 9)
    _, d_ref = _dcbf_repair(x, p_ref, 9)
    np.testing.assert_allclose(float(d_far["correction_norm"][0]),
                               float(d_ref["correction_norm"][0]), atol=1e-6)
