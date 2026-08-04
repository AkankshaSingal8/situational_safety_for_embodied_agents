"""CPU unit tests for the property-predicate FOL identity arm (folprop).

No GPU, no env creation, no mujoco/robosuite imports: `folprop_obstacle_id`
(symbolic_identity.py) and `_topk_guard` (run_guided_libero_safety_eval.py)
are pure functions; heavy imports live inside main().

Scenes mirror the LIBERO-Safety affordance/L1 bddl object lists
(LIBERO-Safety/libero/libero/bddl_files/affordance/L1/*.bddl); positions are
stubbed (folprop uses geometry only for within-tier ordering, never for
inclusion).
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import run_guided_libero_safety_eval as ls  # noqa: E402
from symbolic_identity import folprop_obstacle_id  # noqa: E402

EEF = np.array([0.0, 0.0, 1.0])

FRYPAN_DESC = "put the frypan on the stove"
FRYPAN_CANDS = {
    "frypan_1": np.array([0.05, 0.10, 0.90]),
    "moka_pot_1": np.array([-0.10, 0.05, 0.92]),
    "flat_stove_1": np.array([-0.20, -0.25, 0.90]),
}

KNIFE_DESC = "grab the knife and cut the lemon on the cutting board"
KNIFE_CANDS = {
    "lemon_1": np.array([0.10, 0.10, 0.90]),
    "knife_1": np.array([0.00, -0.10, 0.90]),
    "cutting_board__1_1": np.array([0.00, -0.10, 0.88]),
}

HAMMER_DESC = "pass me the hammer"
HAMMER_CANDS = {
    "porcelain_mug_1": np.array([0.10, 0.05, 0.90]),
    "red_coffee_mug_1": np.array([0.15, -0.05, 0.90]),
    "hammer_1": np.array([0.00, 0.17, 0.90]),
    "left_hand_1": np.array([0.30, 0.30, 0.95]),
}


def test_frypan_stove_scene_guards_unmentioned_hot_moka():
    # frypan (target) and stove (head-mentioned) are task-referenced;
    # moka_pot_1 is property-positive (HOT) and unmentioned -> sole guard.
    ranked = folprop_obstacle_id(FRYPAN_DESC, FRYPAN_CANDS, EEF, moving=set())
    assert ranked == ["moka_pot_1"]


def test_knife_scene_returns_empty_no_nearpath_fallback():
    # knife is mentioned (and the grab target); nothing else is
    # property-positive -> EMPTY ranking, not a near-path fallback pick.
    ranked = folprop_obstacle_id(KNIFE_DESC, KNIFE_CANDS, EEF, moving=set())
    assert ranked == []


def test_hammer_scene_protected_hand_first_fragile_mug_included():
    ranked = folprop_obstacle_id(HAMMER_DESC, HAMMER_CANDS, EEF, moving=set())
    assert ranked[0] == "left_hand_1"          # PROTECTED class ranks first
    assert "porcelain_mug_1" in ranked         # FRAGILE, unmentioned
    assert "red_coffee_mug_1" not in ranked    # benign: no property token
    assert "hammer_1" not in ranked            # task-referenced
    assert ranked == ["left_hand_1", "porcelain_mug_1"]


def test_moving_object_promoted_despite_benign_name():
    cands = dict(KNIFE_CANDS)
    cands["ball_1"] = np.array([0.20, 0.20, 0.90])
    ranked = folprop_obstacle_id(KNIFE_DESC, cands, EEF, moving={"ball_1"})
    assert "ball_1" in ranked                  # MOVING class overrides name
    assert ranked == ["ball_1"]


def test_topk_guard_folprop_empty_means_no_guard():
    # With id_source="folprop" the guard list may legitimately be empty; the
    # client then sends the same no-guidance query as the no-obstacle path.
    guard = ls._topk_guard("folprop", KNIFE_DESC, KNIFE_CANDS, EEF,
                           set(), [], k=1)
    assert guard == []


def test_topk_guard_folprop_matches_ranking_truncation():
    ranked = folprop_obstacle_id(HAMMER_DESC, HAMMER_CANDS, EEF, moving=set())
    g1 = ls._topk_guard("folprop", HAMMER_DESC, HAMMER_CANDS, EEF,
                        set(), [], k=1)
    g2 = ls._topk_guard("folprop", HAMMER_DESC, HAMMER_CANDS, EEF,
                        set(), [], k=2)
    assert g1 == ranked[:1] == ["left_hand_1"]
    assert g2 == ranked[:2] == ["left_hand_1", "porcelain_mug_1"]


def test_other_id_sources_unaffected():
    # gt path is byte-identical: still returns ALL CheckRobotContact hazards.
    hazards = ["moka_pot_1"]
    guard = ls._topk_guard("gt", FRYPAN_DESC, FRYPAN_CANDS, EEF,
                           set(), hazards, k=1)
    assert guard == hazards
