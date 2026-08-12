"""CPU unit tests for `--fol_ablate` (ablation variants of the FOL hazard
election), task-1-brief.md.

Full rule being ablated (symbolic_identity.py):
    HAZARD(x) := PROTECTED(x) | MOVING(x)
                 | (PRED(x) & ~MENTIONED(x) & ~TARGET(x))

Covers `symbolic_identity.ablated_obstacle_id` and its dispatch through
`_topk_guard` in run_guided_libero_safety_eval.py — pure, no GPU, no env
creation, no mujoco/robosuite imports. Modeled on test_fol_predicates.py /
test_folpropvlm_identity.py / test_dynamic_election.py.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import run_guided_libero_safety_eval as ls  # noqa: E402
import symbolic_identity as si  # noqa: E402
from symbolic_identity import (  # noqa: E402
    ablated_obstacle_id,
    fol_obstacle_id,
    folprop_obstacle_id,
    folpropvlm_obstacle_id,
)

EEF = np.array([0.0, 0.0, 1.0])

PROPS = {
    "porcelain mug": {"hot": False, "sharp": False, "fragile": True},
    "moka pot": {"hot": True, "sharp": False, "fragile": False},
    "flat stove": {"hot": True, "sharp": False, "fragile": False},
    "knife": {"hot": False, "sharp": True, "fragile": False},
    "lemon": {"hot": False, "sharp": False, "fragile": False},
    "hammer": {"hot": False, "sharp": False, "fragile": False},
}
HEAT = {
    "flat stove": True,
    "porcelain mug": False,
    "moka pot": False,
    "knife": False,
    "lemon": False,
    "hammer": False,
}


def run_vlm(id_source, ablation, desc, cands, moving=None):
    return ablated_obstacle_id(id_source, ablation, desc, cands, EEF,
                               moving=moving, properties=PROPS,
                               heat_sources=HEAT)


# --- "none" delegation: byte-identical to the incumbent functions ---------

SCENE_HAND_MOVER = {
    "left_hand_1": np.array([0.05, 0.0, 0.95]),
    "moka_pot_1": np.array([0.30, 0.30, 0.92]),
    "lemon_1": np.array([0.10, 0.10, 0.90]),
}
SCENE_PROPERTY = {
    "porcelain_mug_1": np.array([0.10, 0.05, 0.90]),
    "hammer_1": np.array([0.00, 0.17, 0.90]),
}
SCENE_NONE = {
    "lemon_1": np.array([0.10, 0.10, 0.90]),
    "ball_1": np.array([0.20, 0.20, 0.90]),
}


def test_none_delegation_fol():
    for desc, cands, moving in [
        ("pick up the lemon", SCENE_HAND_MOVER, {"left_hand_1"}),
        ("pass me the hammer", SCENE_PROPERTY, None),
        ("pick up the lemon", SCENE_NONE, None),
    ]:
        incumbent = fol_obstacle_id(desc, cands, EEF, moving=moving)
        ablated = ablated_obstacle_id("fol", "none", desc, cands, EEF,
                                      moving=moving)
        assert ablated == incumbent


def test_none_delegation_folprop():
    for desc, cands, moving in [
        ("pick up the lemon", SCENE_HAND_MOVER, {"left_hand_1"}),
        ("pass me the hammer", SCENE_PROPERTY, None),
        ("pick up the lemon", SCENE_NONE, None),
    ]:
        incumbent = folprop_obstacle_id(desc, cands, EEF, moving=moving)
        ablated = ablated_obstacle_id("folprop", "none", desc, cands, EEF,
                                      moving=moving)
        assert ablated == incumbent


def test_none_delegation_folpropvlm():
    for desc, cands, moving in [
        ("pick up the lemon", SCENE_HAND_MOVER, {"left_hand_1"}),
        ("pass me the hammer", SCENE_PROPERTY, None),
        ("pick up the lemon", SCENE_NONE, None),
    ]:
        incumbent = folpropvlm_obstacle_id(desc, cands, EEF, moving=moving,
                                           properties=PROPS, heat_sources=HEAT)
        ablated = run_vlm("folpropvlm", "none", desc, cands, moving=moving)
        assert ablated == incumbent


# --- geom_only --------------------------------------------------------------

def test_geom_only_includes_mentioned_and_target_ranked_by_path():
    # "pick up the lemon": lemon is mentioned/target, but geom_only ranks it
    # in anyway, purely by distance to the eef->target segment.
    cands = {
        "lemon_1": np.array([0.05, 0.0, 0.90]),   # on the path (also target)
        "hammer_1": np.array([5.0, 5.0, 0.90]),   # far off-path
    }
    out = ablated_obstacle_id("fol", "geom_only", "pick up the lemon", cands,
                              EEF)
    assert "lemon_1" in out
    assert set(out) == set(cands)
    tgt = si.parse_target_heuristic("pick up the lemon", list(cands))
    tpos = cands.get(tgt, EEF)
    expected = sorted(cands, key=lambda k: si.segment_distance(cands[k], EEF, tpos))
    assert out == expected


def test_geom_only_same_across_id_sources():
    cands = {
        "lemon_1": np.array([0.05, 0.0, 0.90]),
        "hammer_1": np.array([0.20, 0.10, 0.90]),
    }
    desc = "pick up the lemon"
    out_fol = ablated_obstacle_id("fol", "geom_only", desc, cands, EEF)
    out_folprop = ablated_obstacle_id("folprop", "geom_only", desc, cands, EEF)
    out_folpropvlm = run_vlm("folpropvlm", "geom_only", desc, cands)
    assert out_fol == out_folprop == out_folpropvlm


# --- no_exempt ---------------------------------------------------------------

def test_no_exempt_fol_target_appears_in_ablated_absent_from_incumbent():
    # Target is unmentioned-fallback candidate normally excluded by ~TARGET;
    # no_exempt drops that exemption so it appears in the ranking.
    desc = "pick up the knife"
    cands = {
        "knife_1": np.array([0.05, 0.0, 0.90]),
        "lemon_1": np.array([5.0, 5.0, 0.90]),
    }
    incumbent = fol_obstacle_id(desc, cands, EEF)
    ablated = ablated_obstacle_id("fol", "no_exempt", desc, cands, EEF)
    assert "knife_1" not in incumbent
    assert "knife_1" in ablated


def test_no_exempt_folprop_target_property_positive_elected():
    desc = "pick up the knife"
    cands = {
        "knife_1": np.array([0.05, 0.0, 0.90]),
        "lemon_1": np.array([0.20, 0.10, 0.90]),
    }
    incumbent = folprop_obstacle_id(desc, cands, EEF)
    ablated = ablated_obstacle_id("folprop", "no_exempt", desc, cands, EEF)
    assert "knife_1" not in incumbent
    assert "knife_1" in ablated


def test_no_exempt_folpropvlm_target_property_positive_elected():
    desc = "pick up the knife"
    cands = {
        "knife_1": np.array([0.05, 0.0, 0.90]),
        "lemon_1": np.array([0.20, 0.10, 0.90]),
    }
    incumbent = folpropvlm_obstacle_id(desc, cands, EEF, properties=PROPS,
                                       heat_sources=HEAT)
    ablated = run_vlm("folpropvlm", "no_exempt", desc, cands)
    assert "knife_1" not in incumbent
    assert "knife_1" in ablated


def test_no_exempt_protected_block_unchanged():
    desc = "pick up the lemon"
    cands = dict(SCENE_HAND_MOVER)
    out = ablated_obstacle_id("fol", "no_exempt", desc, cands, EEF,
                              moving={"left_hand_1"})
    assert out[0] == "left_hand_1"


# --- no_class -----------------------------------------------------------

def test_no_class_hand_in_incumbent_absent_from_ablated_fol():
    desc = "pick up the lemon"
    cands = dict(SCENE_HAND_MOVER)  # left_hand near path
    incumbent = fol_obstacle_id(desc, cands, EEF, moving={"left_hand_1"})
    ablated = ablated_obstacle_id("fol", "no_class", desc, cands, EEF,
                                  moving={"left_hand_1"})
    assert "left_hand_1" in incumbent
    assert "left_hand_1" not in ablated


def test_no_class_moving_set_ignored_for_name_protected_entry():
    # A name-protected entry ("hand") is excluded from the no_class result
    # regardless of whether it is ALSO passed in `moving` -- name-token
    # protection already excludes it from the PRED-branch obstacles list,
    # so the moving flag makes no difference for this entry.
    desc = "pick up the lemon"
    cands = dict(SCENE_HAND_MOVER)
    ablated_no_mover = ablated_obstacle_id("fol", "no_class", desc, cands, EEF,
                                           moving=None)
    ablated_with_mover = ablated_obstacle_id("fol", "no_class", desc, cands,
                                             EEF, moving={"left_hand_1"})
    assert ablated_no_mover == ablated_with_mover
    assert "left_hand_1" not in ablated_no_mover
    assert "left_hand_1" not in ablated_with_mover


def test_no_class_folpropvlm_property_positive_unmentioned_nontarget():
    desc = "pass me the hammer"
    cands = {
        "porcelain_mug_1": np.array([0.10, 0.05, 0.90]),
        "left_hand_1": np.array([0.30, 0.30, 0.95]),
    }
    incumbent = run_vlm("folpropvlm", "none", desc, cands,
                        moving={"left_hand_1"})
    ablated = run_vlm("folpropvlm", "no_class", desc, cands,
                      moving={"left_hand_1"})
    assert incumbent[0] == "left_hand_1"
    assert "left_hand_1" not in ablated
    assert ablated == ["porcelain_mug_1"]


# --- no_heat_gate (folpropvlm only) -----------------------------------------

def test_no_heat_gate_hot_without_heat_source_elected_under_ablation():
    desc = "pick up the lemon"
    cands = {
        "moka_pot_1": np.array([-0.10, 0.05, 0.92]),  # HOT, no heat source nearby
        "lemon_1": np.array([0.10, 0.10, 0.90]),
    }
    incumbent = run_vlm("folpropvlm", "none", desc, cands)
    ablated = run_vlm("folpropvlm", "no_heat_gate", desc, cands)
    assert "moka_pot_1" not in incumbent
    assert "moka_pot_1" in ablated


def test_no_heat_gate_noop_for_fol_and_folprop():
    desc = "pick up the lemon"
    cands = dict(SCENE_HAND_MOVER)
    for id_source, incumbent_fn in [
        ("fol", fol_obstacle_id),
        ("folprop", folprop_obstacle_id),
    ]:
        incumbent = incumbent_fn(desc, cands, EEF, moving={"left_hand_1"})
        ablated = ablated_obstacle_id(id_source, "no_heat_gate", desc, cands,
                                      EEF, moving={"left_hand_1"})
        assert ablated == incumbent


# --- unknown ablation --------------------------------------------------------

def test_unknown_ablation_raises_value_error():
    with pytest.raises(ValueError):
        ablated_obstacle_id("fol", "bogus_ablation", "pick up the lemon",
                            SCENE_NONE, EEF)


def test_unknown_id_source_raises_value_error_for_no_exempt():
    with pytest.raises(ValueError):
        ablated_obstacle_id("gt", "no_exempt", "pick up the lemon",
                            SCENE_NONE, EEF)


# --- dispatch-level defaults-off (mirrors test_dynamic_election.py) --------

def test_argparse_fol_ablate_default_none_and_choices():
    src = pathlib.Path(ls.__file__).read_text()
    assert ('"--fol_ablate", choices=["none", "geom_only", "no_exempt", '
            '"no_class", "no_heat_gate"]') in src
    after = src.split('"--fol_ablate"')[1][:200]
    assert 'default="none"' in after


def test_defaults_off_fol_ablate_none_leaves_topk_guard_output_identical():
    # With --fol_ablate unset (argparse default "none"), _topk_guard's
    # dispatch branch for fol_ablate != "none" is never taken, so the
    # output is exactly the incumbent fol_obstacle_id ranking -- byte-
    # identical to pre-Task-1 behavior.
    desc = "pick up the lemon"
    cands = dict(SCENE_HAND_MOVER)
    movers = {"left_hand_1"}
    fol_ablate = "none"  # the argparse default
    guard_default = ls._topk_guard("fol", desc, cands, EEF, movers, [], k=2,
                                   fol_ablate=fol_ablate)
    guard_no_kwarg = ls._topk_guard("fol", desc, cands, EEF, movers, [], k=2)
    incumbent = fol_obstacle_id(desc, cands, EEF, moving=movers)[:2]
    assert guard_default == guard_no_kwarg == incumbent


def test_topk_guard_dispatches_through_ablation_when_set():
    desc = "pick up the lemon"
    cands = dict(SCENE_HAND_MOVER)
    movers = {"left_hand_1"}
    guard = ls._topk_guard("fol", desc, cands, EEF, movers, [], k=2,
                           fol_ablate="no_class")
    ablated = ablated_obstacle_id("fol", "no_class", desc, cands, EEF,
                                  moving=movers)[:2]
    assert guard == ablated
    assert "left_hand_1" not in guard


def test_topk_guard_noop_for_symbolic_and_gt_sources():
    hazards = ["moka_pot_1"]
    cands = {"moka_pot_1": np.array([-0.10, 0.05, 0.92])}
    guard_gt = ls._topk_guard("gt", "put the frypan on the stove", cands,
                              EEF, set(), hazards, k=1, fol_ablate="no_class")
    assert guard_gt == hazards

    guard_symbolic_ablate = ls._topk_guard(
        "symbolic", "pick up the lemon", SCENE_NONE, EEF, set(), [], k=1,
        fol_ablate="geom_only")
    guard_symbolic_none = ls._topk_guard(
        "symbolic", "pick up the lemon", SCENE_NONE, EEF, set(), [], k=1,
        fol_ablate="none")
    assert guard_symbolic_ablate == guard_symbolic_none


def test_results_dict_echoes_fol_ablate_flag():
    src = pathlib.Path(ls.__file__).read_text()
    assert '"fol_ablate": args.fol_ablate' in src
