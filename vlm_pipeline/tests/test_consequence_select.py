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


def test_aggregate_p_any_hazard_mask_ignores_high_target_contact_prob():
    """HIGH fix regression: a candidate with near-certain TARGET contact
    (slot 1, e.g. a successful grasp approach) but near-zero HAZARD contact
    (slot 0) must score as near-feasible when aggregated over hazard_mask
    only -- aggregating over entity_mask (all entities incl. target)
    instead would make this exact progress-making candidate look
    infeasible, which is the mechanism-defeating bug this fix resolves."""
    contact_prob = np.array([[0.01, 0.98]])  # [hazard, target]
    entity_mask = np.array([1.0, 1.0])   # both are valid model-input slots
    hazard_mask = np.array([1.0, 0.0])   # only slot 0 (hazard) counts

    p_any_all_entities = cs.aggregate_p_any(contact_prob, entity_mask)
    p_any_hazard_only = cs.aggregate_p_any(contact_prob, hazard_mask)

    assert p_any_all_entities[0] > 0.9   # would fail feasibility (>0.10)
    assert p_any_hazard_only[0] < 0.05   # passes feasibility (<=0.10)

    res_all = cs.select_candidate(p_any_all_entities, np.zeros(1), np.array([1.0]))
    res_hazard = cs.select_candidate(p_any_hazard_only, np.zeros(1), np.array([1.0]))
    assert res_all.fallback  # bug reproduction: all-entities aggregation -> unsafe
    assert not res_hazard.fallback and res_hazard.chosen == 0  # fixed behavior


# --------------------------------------------------------------------------
# build_features: sorted(entities)[:E] slot construction from a raw
# guidance payload (fix round 1 -- resolved entity-slot contract).
# --------------------------------------------------------------------------

def test_build_features_sorted_entity_slot_order_and_presence_mask():
    payload = {
        "eef_pos": [0.0, 0.0, 0.9],
        "entities": {
            "zeta": [0.0, 0.2, 0.9],   # sorts after hazard_a -> slot 1
            "hazard_a": [0.1, 0.0, 0.9],  # sorts first -> slot 0
        },
        "hazards": ["hazard_a"],
    }
    eef = np.array([0.0, 0.0, 0.9])
    k, extra_h = 2, cs.H + 3
    disp = np.zeros((k, extra_h, 3))
    disp[:, :, 0] = 0.01  # constant x-displacement per step
    feats = cs.build_features(payload, eef, disp, domain="SL")
    assert feats["entity_mask"].shape == (k, cs.E)
    np.testing.assert_array_equal(feats["entity_mask"][0], [1, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(feats["entity_rel"][0, 0], [0.1, 0.0, 0.0])  # hazard_a
    np.testing.assert_allclose(feats["entity_rel"][0, 1], [0.0, 0.2, 0.0])  # zeta
    # hazard_mask is a SUBSET of entity_mask: only hazard_a (slot 0), not
    # zeta (present in entities/entity_mask but not in "hazards" -> the
    # task-target-like slot must NOT be included in hazard_mask).
    assert feats["hazard_mask"].shape == (k, cs.E)
    np.testing.assert_array_equal(feats["hazard_mask"][0], [1, 0, 0, 0, 0, 0, 0, 0])
    assert feats["action_window"].shape == (k, cs.H, 3)  # truncated to H, not extra_h
    assert feats["grip"].shape == (k, 2)
    np.testing.assert_array_equal(feats["grip"], np.zeros((k, 2)))  # gripper_qpos absent


def test_build_features_gripper_qpos_passthrough_when_present():
    payload = {
        "eef_pos": [0.0, 0.0, 0.0],
        "entities": {"hazard_a": [0.1, 0.0, 0.0]},
        "hazards": ["hazard_a"],
        "gripper_qpos": [0.011, 0.013],
    }
    feats = cs.build_features(payload, np.zeros(3), np.zeros((1, cs.H, 3)), domain="SL")
    np.testing.assert_allclose(feats["grip"][0], [0.011, 0.013])


def test_build_features_truncates_beyond_8_entities_alphabetically():
    names = [f"e{i}" for i in range(10)]  # e0..e9 -- already sorted
    entities = {n: [float(i), 0.0, 0.0] for i, n in enumerate(names)}
    payload = {"eef_pos": [0.0, 0.0, 0.0], "entities": entities, "hazards": ["e0"]}
    feats = cs.build_features(payload, np.zeros(3), np.zeros((1, cs.H, 3)), domain="SL")
    assert feats["entity_mask"][0].sum() == cs.E  # only first 8 populated
    for slot in range(cs.E):
        np.testing.assert_allclose(feats["entity_rel"][0, slot], [float(slot), 0.0, 0.0])


def test_build_features_hazard_beyond_truncation_raises_hazards_missing_error():
    """A hazard name that only exists past the E=8 truncation point (e.g.
    e9, sorted last) can never be matched to a slot -> hard-fail, not a
    silent 'aggregate over what happened to survive truncation'."""
    names = [f"e{i}" for i in range(10)]
    entities = {n: [float(i), 0.0, 0.0] for i, n in enumerate(names)}
    payload = {"eef_pos": [0.0, 0.0, 0.0], "entities": entities, "hazards": ["e9"]}
    with pytest.raises(cs.HazardsMissingError):
        cs.build_features(payload, np.zeros(3), np.zeros((1, cs.H, 3)), domain="SL")


def test_build_features_missing_entities_raises_entities_missing_error():
    payload = {"eef_pos": [0.0, 0.0, 0.0]}  # no "entities" key
    with pytest.raises(cs.EntitiesMissingError):
        cs.build_features(payload, np.zeros(3), np.zeros((1, cs.H, 3)), domain="SL")


def test_build_features_empty_entities_dict_raises_entities_missing_error():
    payload = {"eef_pos": [0.0, 0.0, 0.0], "entities": {}}
    with pytest.raises(cs.EntitiesMissingError):
        cs.build_features(payload, np.zeros(3), np.zeros((1, cs.H, 3)), domain="SL")


def test_build_features_missing_hazards_raises_hazards_missing_error():
    """entities present but no 'hazards' key -> hard fail, NOT a silent
    aggregate-over-all-entities (which would include the target)."""
    payload = {"eef_pos": [0.0, 0.0, 0.0], "entities": {"hazard_a": [0.1, 0.0, 0.0]}}
    with pytest.raises(cs.HazardsMissingError):
        cs.build_features(payload, np.zeros(3), np.zeros((1, cs.H, 3)), domain="SL")


def test_build_features_empty_hazards_list_raises_hazards_missing_error():
    payload = {"eef_pos": [0.0, 0.0, 0.0],
               "entities": {"hazard_a": [0.1, 0.0, 0.0]}, "hazards": []}
    with pytest.raises(cs.HazardsMissingError):
        cs.build_features(payload, np.zeros(3), np.zeros((1, cs.H, 3)), domain="SL")


def test_build_features_hazards_not_subset_of_entities_raises_hazards_missing_error():
    """None of the given hazard names match an entity slot -> hard fail
    (server-side defensive check; the client is also expected to log-warn
    on this, but the server cannot trust the client)."""
    payload = {"eef_pos": [0.0, 0.0, 0.0],
               "entities": {"hazard_a": [0.1, 0.0, 0.0]},
               "hazards": ["typo_name_not_in_entities"]}
    with pytest.raises(cs.HazardsMissingError):
        cs.build_features(payload, np.zeros(3), np.zeros((1, cs.H, 3)), domain="SL")


def test_build_features_hazard_mask_excludes_target_like_entity():
    """The core HIGH-fix regression test: a target-like entity present in
    'entities' but absent from 'hazards' must NOT appear in hazard_mask,
    even though it's a fully valid (masked-in) entity_mask slot for the
    model's own forward pass."""
    payload = {
        "eef_pos": [0.0, 0.0, 0.0],
        "entities": {"hazard_a": [0.1, 0.0, 0.0], "task_target": [0.2, 0.0, 0.0]},
        "hazards": ["hazard_a"],
    }
    feats = cs.build_features(payload, np.zeros(3), np.zeros((1, cs.H, 3)), domain="SL")
    names = sorted(payload["entities"].keys())  # ["hazard_a", "task_target"]
    hazard_slot = names.index("hazard_a")
    target_slot = names.index("task_target")
    assert feats["entity_mask"][0, hazard_slot] == 1
    assert feats["entity_mask"][0, target_slot] == 1  # target still a valid model input
    assert feats["hazard_mask"][0, hazard_slot] == 1
    assert feats["hazard_mask"][0, target_slot] == 0  # but NOT a hazard-aggregation slot


def test_build_features_action_scale_and_clip():
    payload = {"eef_pos": [0.0, 0.0, 0.0], "entities": {"hazard_a": [0.1, 0.0, 0.0]},
               "hazards": ["hazard_a"]}
    eef = np.zeros(3)
    disp = np.ones((1, cs.H, 3)) * 0.5
    feats = cs.build_features(payload, eef, disp, domain="SL", action_scale=2.0, action_clip=0.6)
    # 0.5 * 2.0 = 1.0, clipped to 0.6
    np.testing.assert_allclose(feats["action_window"], np.full((1, cs.H, 3), 0.6))


def test_build_features_pads_short_window():
    payload = {"entities": {"hazard_a": [0.0, 0.0, 0.0]}, "hazards": ["hazard_a"]}
    eef = np.zeros(3)
    disp = np.ones((1, cs.H - 3, 3))  # shorter than H
    feats = cs.build_features(payload, eef, disp, domain="LS")
    assert feats["action_window"].shape == (1, cs.H, 3)
    np.testing.assert_allclose(feats["action_window"][0, -3:], 0.0)


def test_build_features_unknown_domain_raises():
    payload = {"entities": {"hazard_a": [0.0, 0.0, 0.0]}, "hazards": ["hazard_a"]}
    with pytest.raises(ValueError):
        cs.build_features(payload, np.zeros(3), np.zeros((1, cs.H, 3)), domain="XX")


# --------------------------------------------------------------------------
# Train/inference parity: cs.build_features must reproduce EXACTLY the
# entity_rel/entity_mask/slot-order that cm.build_samples computed for the
# equivalent (eef, entities) state -- the binding contract this fix round
# resolves. (fix round 1)
# --------------------------------------------------------------------------

def test_build_features_matches_build_samples_entity_rel_and_mask():
    # One synthetic episode, entities named so alphabetical sort is
    # non-trivial (not already insertion order).
    eef_start = (0.0, 0.0, 0.9)
    entity_positions = {
        "zzz_far": (0.5, 0.5, 0.9),
        "aaa_hazard": (0.05, 0.0, 0.9),
        "mid_one": (0.2, 0.1, 0.9),
    }
    n_steps = cs.H + 2
    lines = []
    eef = np.array(eef_start, dtype=np.float64)
    for t in range(n_steps):
        eef = eef + np.array([0.001, 0.0, 0.0])
        rec = {
            "task": 0, "ep": 0, "t": t,
            "eef": eef.tolist(), "grip": [0.02, 0.02],
            "action": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            "entities": {n: list(p) for n, p in entity_positions.items()},
        }
        lines.append(rec)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "transitions_obstacle_avoidance_L0.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
        episodes = cm.load_episodes(pathlib.Path(td))
    assert len(episodes) == 1
    samples = cm.build_samples(episodes, target_map=None)
    assert len(samples) >= 1
    s = samples[0]  # anchored at record index 1 -> state = record[0]

    anchor_eef = np.asarray(lines[0]["eef"], dtype=np.float64)
    payload = {
        "eef_pos": anchor_eef.tolist(),
        "entities": {n: list(p) for n, p in entity_positions.items()},
        "hazards": ["aaa_hazard"],  # only one of the three is the "hazard"
    }
    disp = np.zeros((1, cs.H, 3))  # content doesn't matter for this parity check
    feats = cs.build_features(payload, anchor_eef, disp, domain="LS")

    np.testing.assert_allclose(feats["entity_mask"][0], s.entity_mask)
    np.testing.assert_allclose(feats["entity_rel"][0], s.entity_rel, atol=1e-9)
    # Slot order itself: consequence_model's own per-episode entity_order().
    expected_order = sorted(entity_positions.keys())
    assert s.entity_names[: len(expected_order)] == expected_order
    # hazard_mask is the fix-round-2 addition: entity_mask (all 3 present)
    # vs. hazard_mask (only "aaa_hazard", slot index from the SAME sorted
    # order) must differ.
    hazard_slot = expected_order.index("aaa_hazard")
    assert feats["hazard_mask"][0].sum() == 1
    assert feats["hazard_mask"][0, hazard_slot] == 1


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
    legacy dead-code pseudo-slot names in this module ("obstacle",
    "obstacle2") are payload KEYS (pre-existing in guided_policy.py), not a
    name-substring check -- verify no literal `_obstacle_` substring
    appears anywhere in the selection code."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "consequence_select.py").read_text()
    assert "_obstacle_" not in src


def test_guided_policy_hard_fails_missing_entities_into_fallback():
    """Fix round 1/2: a payload without entities OR without matching
    hazards must be caught (EntitiesMissingError / HazardsMissingError) and
    routed to the legacy fallback with a logged warning -- never silently
    degraded (old fixed-slot mapping, or all-entities aggregation)."""
    src = (_REPO_ROOT / "openpi_guided" / "src" / "openpi" / "policies" / "guided_policy.py").read_text()
    assert "except (self._cseq.EntitiesMissingError, self._cseq.HazardsMissingError) as e:" in src
    assert "logging.warning(" in src


def test_run_ensemble_select_aggregates_over_hazard_mask_not_entity_mask():
    """HIGH fix round 2 source check: run_ensemble_select must call
    aggregate_p_any with hazard_mask, not entity_mask (the mechanism-
    defeating bug the fix resolves)."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "consequence_select.py").read_text()
    fn_start = src.index("def run_ensemble_select(")
    fn_end = src.index("\ndef ", fn_start + 1) if "\ndef " in src[fn_start + 1:] else len(src)
    body = src[fn_start:fn_end]
    assert "aggregate_p_any(contact_prob, hazard_mask)" in body
    assert "aggregate_p_any(contact_prob, entity_mask)" not in body


def test_both_clients_attach_hazards_key():
    """Fix round 2: both clients must attach payload['hazards'] alongside
    'entities', and log-warn (not hard-assert) when a hazard name isn't a
    subset of the entities dict."""
    for fname in ("run_guided_libero_safety_eval.py", "run_guided_safelibero_pi05_eval.py"):
        src = (_REPO_ROOT / "vlm_pipeline" / fname).read_text()
        assert 'element["guidance"]["hazards"]' in src
        assert "[consequence]" in src and "not found in" in src  # subset-validation log-warn


def test_build_features_does_not_silently_use_legacy_pseudo_slots():
    """The old fixed-slot helper must be retained only as unused dead code
    (never called by `build_features`)."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "consequence_select.py").read_text()
    assert "_build_features_legacy_pseudo_slots" in src  # retained for reference
    # build_features's own body must not call it.
    fn_start = src.index("def build_features(")
    fn_end = src.index("\ndef ", fn_start + 1)
    assert "_build_features_legacy_pseudo_slots" not in src[fn_start:fn_end]


def test_both_clients_use_one_shared_entity_resolution_helper():
    """Fix round 1: the training-data source (--log_transitions) and the
    inference-time guidance payload must be built from ONE shared
    per-client helper so they can never drift apart."""
    for fname in ("run_guided_libero_safety_eval.py", "run_guided_safelibero_pi05_eval.py"):
        src = (_REPO_ROOT / "vlm_pipeline" / fname).read_text()
        assert "def _resolve_entities(" in src
        # Called at least twice: once for the guidance payload, once for
        # --log_transitions logging.
        assert src.count("_resolve_entities(") >= 3  # def + >=2 call sites
        assert 'element["guidance"]["entities"]' in src
        assert 'element["guidance"]["gripper_qpos"]' in src


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
        "entities": {"hazard_a": [0.03, 0.0, 0.9]},
        "hazards": ["hazard_a"],
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
    payload = {"eef_pos": [0.0, 0.0, 0.9], "entities": {"hazard_a": [0.03, 0.0, 0.9]},
               "hazards": ["hazard_a"]}
    k = 8
    disp = np.random.default_rng(0).normal(size=(k, cs.H, 3)) * 0.01
    feats = cs.build_features(payload, np.array([0.0, 0.0, 0.9]), disp, domain="LS")

    cs.run_ensemble_select(models, norm, feats)  # warm-up

    t0 = time.perf_counter()
    cs.run_ensemble_select(models, norm, feats)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 10.0, f"consequence ensemble forward took {elapsed_ms:.2f} ms (budget: 10 ms)"
