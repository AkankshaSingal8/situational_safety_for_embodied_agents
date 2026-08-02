"""CPU tests for `consequence_select.py` (Task 3, consequence-steering-code
plan, spec 2026-08-01) -- the `--consequence_select` selection layer.

Pure-numpy selection-rule tests always run. Ensemble tests (checkpoint
load/inference/latency) are guarded with `pytest.importorskip("torch")` --
they build a real tiny checkpoint via `consequence_model.train_ensemble` (the
Task 2 helper) rather than hand-rolling a `.pt` file, so the format is
guaranteed compatible with `consequence_model.load_ensemble`. The server-side
JAX wiring (`guided_policy.py`, `pi0_guided.py`) is checked at the source
level only -- this env has no JAX/GPU guarantee and the brief only requires
the selection helper itself to be a pure, JAX/GPU-free importable function.
"""

import json
import pathlib
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import consequence_select as cs  # noqa: E402
import consequence_model as cm  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Pure selection rule: feasibility incl. pessimism, argmax progress, fallback
# --------------------------------------------------------------------------

def test_feasible_includes_pessimism_term():
    # p_mean alone would be "safe" (<=0.10), but +1.0*std pushes it over.
    p_mean = np.array([0.05])
    p_std = np.array([0.10])  # 0.05 + 1.0*0.10 = 0.15 > 0.10
    progress = np.array([1.0])
    res = cs.select_candidate(p_mean, p_std, progress, pessimism=1.0, threshold=0.10)
    assert not res.feasible_mask[0]
    assert res.fallback and res.chosen is None


def test_feasible_boundary_is_inclusive():
    p_mean = np.array([0.06])
    p_std = np.array([0.04])  # 0.06 + 0.04 = 0.10 exactly == threshold -> feasible
    progress = np.array([1.0])
    res = cs.select_candidate(p_mean, p_std, progress, pessimism=1.0, threshold=0.10)
    assert res.feasible_mask[0]
    assert res.chosen == 0
    assert not res.fallback


def test_pessimism_scaling_changes_feasibility():
    p_mean = np.array([0.08])
    p_std = np.array([0.05])
    # lambda=0: feasible (0.08 <= 0.10). lambda=1: infeasible (0.13 > 0.10).
    res0 = cs.select_candidate(p_mean, p_std, np.array([1.0]), pessimism=0.0, threshold=0.10)
    res1 = cs.select_candidate(p_mean, p_std, np.array([1.0]), pessimism=1.0, threshold=0.10)
    assert res0.feasible_mask[0] and not res0.fallback
    assert not res1.feasible_mask[0] and res1.fallback


def test_argmax_progress_among_feasible_ignores_infeasible_higher_progress():
    # Candidate 1 has the highest raw progress but is infeasible; candidate 0
    # is feasible with lower progress than 1 but higher than the other
    # feasible candidate (2) -> must choose 0, not 1.
    p_mean = np.array([0.03, 0.03, 0.03])
    p_std = np.array([0.02, 0.20, 0.02])  # candidate 1: 0.03+0.20=0.23 infeasible
    progress = np.array([5.0, 100.0, 1.0])
    res = cs.select_candidate(p_mean, p_std, progress, pessimism=1.0, threshold=0.10)
    assert list(res.feasible_mask) == [True, False, True]
    assert res.feasible_count == 2
    assert res.chosen == 0
    assert not res.fallback


def test_empty_feasible_set_flags_fallback():
    p_mean = np.array([0.5, 0.6, 0.9])
    p_std = np.array([0.0, 0.0, 0.0])
    progress = np.array([1.0, 2.0, 3.0])
    res = cs.select_candidate(p_mean, p_std, progress, pessimism=1.0, threshold=0.10)
    assert res.feasible_count == 0
    assert res.chosen is None
    assert res.fallback
    d = res.log_dict()
    assert d == {"feasible": 0, "chosen": None, "fallback": True}


def test_log_dict_shape_matches_brief():
    p_mean = np.array([0.01])
    p_std = np.array([0.01])
    progress = np.array([1.0])
    res = cs.select_candidate(p_mean, p_std, progress)
    d = res.log_dict()
    assert set(d.keys()) == {"feasible", "chosen", "fallback"}
    assert d["feasible"] == 1 and d["chosen"] == 0 and d["fallback"] is False


def test_select_candidate_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cs.select_candidate(np.zeros(3), np.zeros(2), np.zeros(3))


# --------------------------------------------------------------------------
# aggregate_p_any
# --------------------------------------------------------------------------

def test_aggregate_p_any_excludes_masked_slots():
    # 2 candidates, E=3 slots; slot 2 is absent (mask=0) for both -> its
    # (overconfident) prob must not affect p_any.
    contact_prob = np.array([
        [0.5, 0.0, 0.99],
        [0.0, 0.0, 0.99],
    ])
    mask = np.array([1.0, 1.0, 0.0])
    p_any = cs.aggregate_p_any(contact_prob, mask)
    assert np.isclose(p_any[0], 1 - (1 - 0.5) * (1 - 0.0))
    assert np.isclose(p_any[1], 0.0)


def test_aggregate_p_any_all_masked_is_zero():
    contact_prob = np.array([[0.9, 0.9]])
    mask = np.array([0.0, 0.0])
    p_any = cs.aggregate_p_any(contact_prob, mask)
    assert np.isclose(p_any[0], 0.0)


# --------------------------------------------------------------------------
# build_features: fixed pseudo-slot construction from a raw guidance payload
# --------------------------------------------------------------------------

def test_build_features_fixed_slot_order_and_presence_mask():
    payload = {
        "eef_pos": [0.0, 0.0, 0.9],
        "obstacle_pos": [0.1, 0.0, 0.9],
        # obstacle2 absent -> slot 1 masked
        "target_pos": [0.0, 0.2, 0.9],
        # dest_pos absent -> slot 3 masked
    }
    eef = np.array([0.0, 0.0, 0.9])
    k, extra_h = 2, cs.H + 3
    disp = np.zeros((k, extra_h, 3))
    disp[:, :, 0] = 0.01  # constant x-displacement per step
    feats = cs.build_features(payload, eef, disp, domain="SL")
    assert feats["entity_mask"].shape == (k, cs.E)
    np.testing.assert_array_equal(feats["entity_mask"][0], [1, 0, 1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(feats["entity_rel"][0, 0], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(feats["entity_rel"][0, 2], [0.0, 0.2, 0.0])
    assert feats["action_window"].shape == (k, cs.H, 3)  # truncated to H, not extra_h
    assert feats["grip"].shape == (k, 2)
    np.testing.assert_array_equal(feats["grip"], np.zeros((k, 2)))  # documented degradation


def test_build_features_action_scale_and_clip():
    payload = {"eef_pos": [0.0, 0.0, 0.0]}
    eef = np.zeros(3)
    disp = np.ones((1, cs.H, 3)) * 0.5
    feats = cs.build_features(payload, eef, disp, domain="SL", action_scale=2.0, action_clip=0.6)
    # 0.5 * 2.0 = 1.0, clipped to 0.6
    np.testing.assert_allclose(feats["action_window"], np.full((1, cs.H, 3), 0.6))


def test_build_features_pads_short_window():
    payload = {}
    eef = np.zeros(3)
    disp = np.ones((1, cs.H - 3, 3))  # shorter than H
    feats = cs.build_features(payload, eef, disp, domain="LS")
    assert feats["action_window"].shape == (1, cs.H, 3)
    np.testing.assert_allclose(feats["action_window"][0, -3:], 0.0)


def test_build_features_unknown_domain_raises():
    with pytest.raises(ValueError):
        cs.build_features({}, np.zeros(3), np.zeros((1, cs.H, 3)), domain="XX")


# --------------------------------------------------------------------------
# defaults-off (None) source-level wiring check -- server files
# --------------------------------------------------------------------------

def test_guided_policy_source_gates_consequence_path_on_flag():
    src = (_REPO_ROOT / "openpi_guided" / "src" / "openpi" / "policies" / "guided_policy.py").read_text()
    assert "consequence_select: str | None = None" in src
    # The consequence ensemble must only be loaded inside an `if
    # config.consequence_select:` guard -- default None is falsy, so the
    # torch-dependent branch (and the extra jit compile) is never reached
    # when the flag is off.
    assert "if config.consequence_select:" in src
    assert "self._consequence_models = None" in src
    # infer() must only route to the consequence path when models were
    # actually loaded (elif self._consequence_models is not None), so the
    # off path falls through to the pre-existing `self._sample_actions` call
    # exactly as before this change.
    assert "elif self._consequence_models is not None:" in src


def test_serve_policy_guided_source_exposes_flag_default_none():
    src = (_REPO_ROOT / "openpi_guided" / "scripts" / "serve_policy_guided.py").read_text()
    assert "consequence_select: str | None = None" in src


def test_pi0_guided_return_candidates_defaults_false_and_preserves_2_tuple():
    src = (_REPO_ROOT / "openpi_guided" / "src" / "openpi" / "models" / "pi0_guided.py").read_text()
    assert "return_candidates: bool = False" in src
    assert 'return selected, diag' in src  # legacy 2-tuple path still present verbatim


def test_no_obstacle_name_substring_used_in_selection_logic():
    """Global constraint: never use `_obstacle_` in decision logic. The
    fixed pseudo-slot names in this module ("obstacle", "obstacle2") are
    payload KEYS (pre-existing in guided_policy.py), not a name-substring
    check -- verify no literal `_obstacle_` substring appears anywhere in
    the new selection code."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "consequence_select.py").read_text()
    assert "_obstacle_" not in src


# --------------------------------------------------------------------------
# Ensemble load/inference (torch-guarded)
# --------------------------------------------------------------------------

def _write_episode(path, task, ep, n_steps, entity_name, entity_pos, eef_start, action_xyz, grip_open):
    eef = np.array(eef_start, dtype=np.float64)
    entity = np.array(entity_pos, dtype=np.float64)
    lines = []
    for t in range(n_steps):
        eef = eef + np.array(action_xyz)
        rec = {
            "task": task, "ep": ep, "t": t,
            "eef": eef.tolist(), "grip": list(grip_open),
            "action": list(action_xyz) + [0.0, 0.0, 0.0, -1.0],
            "entities": {entity_name: entity.tolist()},
        }
        lines.append(json.dumps(rec))
    existing = path.read_text() if path.exists() else ""
    path.write_text(existing + "\n".join(lines) + "\n")


@pytest.fixture
def tiny_ensemble_dir(tmp_path):
    pytest.importorskip("torch")
    d = tmp_path / "data"
    d.mkdir()
    p = d / "transitions_obstacle_avoidance_L0.jsonl"
    for ep in range(6):
        _write_episode(
            p, task=0, ep=ep, n_steps=16,
            entity_name="hazard_a", entity_pos=(0.03, 0.0, 0.9),
            eef_start=(0.0, 0.0, 0.9), action_xyz=(0.002, 0.0, 0.0),
            grip_open=(0.02, 0.02),
        )
    episodes = cm.load_episodes(d)
    train_eps = [e for e in episodes if e.ep < 4]
    held_eps = [e for e in episodes if e.ep >= 4]
    target_map = {0: "hazard_a"}
    train_samples = cm.build_samples(train_eps, target_map)
    held_samples = cm.build_samples(held_eps, target_map)
    norm = cm.compute_norm_stats(train_samples)
    out_dir = tmp_path / "ckpt"
    cm.train_ensemble(train_samples, held_samples, norm, out_dir,
                       n_members=3, max_epochs=2, patience=5)
    return out_dir


def test_load_ensemble_for_server_roundtrip(tiny_ensemble_dir):
    models, norm = cs.load_ensemble_for_server(str(tiny_ensemble_dir), n_members=3)
    assert len(models) == 3
    for m in models:
        assert m.training is False  # eval mode


def test_load_ensemble_for_server_missing_dir_raises_clear_error(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(RuntimeError) as excinfo:
        cs.load_ensemble_for_server(str(missing), n_members=3)
    msg = str(excinfo.value)
    assert str(missing) in msg  # names the offending MODEL_DIR
    assert "consequence_select" in msg or "consequence" in msg.lower()


def test_run_ensemble_select_end_to_end(tiny_ensemble_dir):
    models, norm = cs.load_ensemble_for_server(str(tiny_ensemble_dir), n_members=3)
    payload = {
        "eef_pos": [0.0, 0.0, 0.9],
        "obstacle_pos": [0.03, 0.0, 0.9],
    }
    k = 4
    disp = np.zeros((k, cs.H, 3))
    disp[:, :, 0] = np.linspace(0.001, 0.004, k)[:, None]
    feats = cs.build_features(payload, np.array([0.0, 0.0, 0.9]), disp, domain="LS")
    result = cs.run_ensemble_select(models, norm, feats, pessimism=1.0, threshold=0.10)
    assert result.p_mean.shape == (k,)
    assert result.p_std.shape == (k,)
    assert result.progress_mean.shape == (k,)
    assert isinstance(result.fallback, bool)
    if not result.fallback:
        assert 0 <= result.chosen < k
        assert result.feasible_mask[result.chosen]


def test_run_ensemble_select_latency_under_10ms(tiny_ensemble_dir):
    """Structural latency check per the brief: single batched forward for
    K=8, tiny dims, asserted < 10 ms on CPU. Not a benchmark -- a smoke
    bound; warms up once (first call pays torch/JIT-free import overhead)."""
    models, norm = cs.load_ensemble_for_server(str(tiny_ensemble_dir), n_members=3)
    payload = {"eef_pos": [0.0, 0.0, 0.9], "obstacle_pos": [0.03, 0.0, 0.9]}
    k = 8
    disp = np.random.default_rng(0).normal(size=(k, cs.H, 3)) * 0.01
    feats = cs.build_features(payload, np.array([0.0, 0.0, 0.9]), disp, domain="LS")

    cs.run_ensemble_select(models, norm, feats)  # warm-up

    t0 = time.perf_counter()
    cs.run_ensemble_select(models, norm, feats)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 10.0, f"consequence ensemble forward took {elapsed_ms:.2f} ms (budget: 10 ms)"
