import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.traj import (CARRY, K, A_MARGIN, _dense_xy, base_trajectory, laptop_clearance,  # noqa: E402
                      safe_A, safe_AB, safe_B, sample_scene)
from src.texts import TRAIN_A, TRAIN_B, HELDOUT_A, HELDOUT_B, COMBINED  # noqa: E402


def _scenes(n=30, seed=123):
    rng = np.random.default_rng(seed)
    return [sample_scene(rng) for _ in range(n)]


def test_base_violates_both_geometrically():
    for sc in _scenes():
        b = base_trajectory(sc)
        k0, k1 = CARRY
        assert laptop_clearance(_dense_xy(b[k0:k1 + 1, :2]), sc).min() < 0          # passes over laptop
        assert np.degrees(np.linalg.norm(b[:, 3:], axis=-1).max()) > 30              # tilts the mug


def test_oracles_fix_only_their_constraint():
    for sc in _scenes():
        b = base_trajectory(sc)
        a, bb, ab = safe_A(sc, b), safe_B(sc, b), safe_AB(sc, b)
        k0, k1 = CARRY
        assert laptop_clearance(_dense_xy(a[k0:k1 + 1, :2]), sc).min() > A_MARGIN
        np.testing.assert_allclose(a[:, 3:], b[:, 3:])                               # A leaves tilt alone
        np.testing.assert_allclose(bb[:, :3], b[:, :3])                              # B leaves path alone
        assert np.abs(bb[:, 3:]).max() == 0
        np.testing.assert_allclose(ab[:, :3], a[:, :3]); assert np.abs(ab[:, 3:]).max() == 0
        assert a.shape == (K, 6)


def test_detour_side_depends_on_scene():
    sides = [np.sign(safe_A(sc, base_trajectory(sc))[20, 1] - base_trajectory(sc)[20, 1]) for sc in _scenes(60)]
    assert 0.2 < np.mean(np.array(sides) > 0) < 0.8


def test_text_splits_disjoint():
    train = set(TRAIN_A + TRAIN_B)
    assert not train & set(HELDOUT_A + HELDOUT_B)
    assert not train & {c[0] for c in COMBINED}
