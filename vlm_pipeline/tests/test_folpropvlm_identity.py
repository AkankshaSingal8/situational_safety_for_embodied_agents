"""CPU unit tests for the VLM-grounded property FOL identity arm (folpropvlm).

No GPU, no env creation, no mujoco/robosuite imports: `folpropvlm_obstacle_id`
(symbolic_identity.py) and `_topk_guard` (run_guided_libero_safety_eval.py)
are pure functions; heavy imports live inside main().

Property/heat-source tables are injected via the `properties`/`heat_sources`
kwargs — the tests do NOT depend on the real results_tables/ JSONs. Table
keys use the base-class normalization (instance-suffix digits stripped,
underscores -> spaces, lowercase).
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import run_guided_libero_safety_eval as ls  # noqa: E402
from symbolic_identity import folpropvlm_obstacle_id  # noqa: E402

EEF = np.array([0.0, 0.0, 1.0])

PROPS = {
    "porcelain mug": {"hot": False, "sharp": False, "fragile": True},
    "red coffee mug": {"hot": False, "sharp": False, "fragile": False},
    "moka pot": {"hot": True, "sharp": False, "fragile": False},
    "frypan": {"hot": True, "sharp": False, "fragile": False},
    "flat stove": {"hot": True, "sharp": False, "fragile": False},
    "knife": {"hot": False, "sharp": True, "fragile": False},
    "lemon": {"hot": False, "sharp": False, "fragile": False},
    "hammer": {"hot": False, "sharp": False, "fragile": False},
    "ball": {"hot": False, "sharp": False, "fragile": False},
}
HEAT = {
    "flat stove": True,
    "porcelain mug": False,
    "red coffee mug": False,
    "moka pot": False,
    "frypan": False,
    "knife": False,
    "lemon": False,
    "hammer": False,
    "ball": False,
}


def run(desc, cands, moving=None):
    return folpropvlm_obstacle_id(desc, cands, EEF, moving=moving or set(),
                                  properties=PROPS, heat_sources=HEAT)


def test_fragile_unmentioned_guarded_head_mentioned_excluded():
    cands = {
        "porcelain_mug_1": np.array([0.10, 0.05, 0.90]),
        "hammer_1": np.array([0.00, 0.17, 0.90]),
    }
    assert run("pass me the hammer", cands) == ["porcelain_mug_1"]
    # Same fragile object head-mentioned in the instruction -> not guarded.
    assert run("pass me the porcelain mug", cands) == []


def test_hot_class_without_heat_source_not_guarded():
    # moka pot is HOT by class but no heat source exists in the scene ->
    # state-aware rule says it is cold -> NOT guarded.
    cands = {
        "moka_pot_1": np.array([-0.10, 0.05, 0.92]),
        "lemon_1": np.array([0.10, 0.10, 0.90]),
    }
    assert run("pick up the lemon", cands) == []


def test_hot_class_near_heat_source_guarded():
    # Same moka pot within 0.15 m of a heat-source-positive candidate ->
    # effectively hot -> guarded. The stove itself is head-mentioned (goal
    # destination) so only the moka pot is returned.
    cands = {
        "moka_pot_1": np.array([-0.15, -0.20, 0.92]),
        "flat_stove_1": np.array([-0.20, -0.25, 0.90]),
        "lemon_1": np.array([0.10, 0.10, 0.90]),
    }
    assert run("put the lemon on the stove", cands) == ["moka_pot_1"]


def test_hot_class_far_from_heat_source_not_guarded():
    cands = {
        "moka_pot_1": np.array([0.30, 0.30, 0.92]),   # ~0.9 m from stove
        "flat_stove_1": np.array([-0.20, -0.25, 0.90]),
        "lemon_1": np.array([0.10, 0.10, 0.90]),
    }
    assert run("put the lemon on the stove", cands) == []


def test_heat_source_itself_guarded():
    # An unmentioned heat source is a hazard regardless of proximity logic.
    cands = {
        "flat_stove_1": np.array([-0.20, -0.25, 0.90]),
        "lemon_1": np.array([0.10, 0.10, 0.90]),
    }
    assert run("pick up the lemon", cands) == ["flat_stove_1"]


def test_protected_hand_always_first():
    cands = {
        "porcelain_mug_1": np.array([0.10, 0.05, 0.90]),
        "hammer_1": np.array([0.00, 0.17, 0.90]),
        "left_hand_1": np.array([0.30, 0.30, 0.95]),
    }
    ranked = run("pass me the hammer", cands)
    assert ranked[0] == "left_hand_1"          # PROTECTED class ranks first
    assert ranked == ["left_hand_1", "porcelain_mug_1"]


def test_moving_object_promoted_despite_negative_properties():
    cands = {
        "lemon_1": np.array([0.10, 0.10, 0.90]),
        "ball_1": np.array([0.20, 0.20, 0.90]),
    }
    ranked = run("pick up the lemon", cands, moving={"ball_1"})
    assert ranked == ["ball_1"]                # MOVING class overrides tables


def test_empty_when_nothing_qualifies():
    cands = {
        "lemon_1": np.array([0.10, 0.10, 0.90]),
        "hammer_1": np.array([0.00, 0.17, 0.90]),
    }
    assert run("pick up the lemon", cands) == []


def test_unknown_class_absent_from_tables_not_guarded():
    cands = {
        "mystery_widget_1": np.array([0.10, 0.05, 0.90]),
        "lemon_1": np.array([0.10, 0.10, 0.90]),
    }
    assert run("pick up the lemon", cands) == []


def test_topk_guard_folpropvlm_dispatch_and_truncation(monkeypatch):
    # Exercise the eval-runner dispatch without touching the real
    # results_tables files: pre-seed the lazy module-level cache with the
    # test tables (the loader short-circuits when the cache is populated).
    import symbolic_identity as si
    monkeypatch.setattr(si, "_VLM_PROP_TABLE", PROPS)
    monkeypatch.setattr(si, "_VLM_HEAT_TABLE", HEAT)
    cands = {
        "porcelain_mug_1": np.array([0.10, 0.05, 0.90]),
        "hammer_1": np.array([0.00, 0.17, 0.90]),
        "left_hand_1": np.array([0.30, 0.30, 0.95]),
    }
    ranked = run("pass me the hammer", cands)
    g1 = ls._topk_guard("folpropvlm", "pass me the hammer", cands, EEF,
                        set(), [], k=1)
    g2 = ls._topk_guard("folpropvlm", "pass me the hammer", cands, EEF,
                        set(), [], k=2)
    assert g1 == ranked[:1] == ["left_hand_1"]
    assert g2 == ranked[:2] == ["left_hand_1", "porcelain_mug_1"]


def test_topk_guard_folpropvlm_empty_means_no_guard(monkeypatch):
    import symbolic_identity as si
    monkeypatch.setattr(si, "_VLM_PROP_TABLE", PROPS)
    monkeypatch.setattr(si, "_VLM_HEAT_TABLE", HEAT)
    cands = {
        "lemon_1": np.array([0.10, 0.10, 0.90]),
        "hammer_1": np.array([0.00, 0.17, 0.90]),
    }
    guard = ls._topk_guard("folpropvlm", "pick up the lemon", cands, EEF,
                           set(), [], k=1)
    assert guard == []


def test_topk_guard_other_sources_unaffected():
    # gt path is byte-identical: still returns ALL CheckRobotContact hazards.
    hazards = ["moka_pot_1"]
    cands = {"moka_pot_1": np.array([-0.10, 0.05, 0.92])}
    guard = ls._topk_guard("gt", "put the frypan on the stove", cands, EEF,
                           set(), hazards, k=1)
    assert guard == hazards
