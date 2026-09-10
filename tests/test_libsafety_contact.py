"""Tests for the independent robot-hazard contact scorer.

The benchmark's own CheckRobotContact (bddl_base_domain.py:1057) appends geom
INDICES to its group while _check_contact resolves contacts to NAMES and tests
membership, so it can never return True. These tests pin the corrected
behaviour: work in index space throughout.
"""
from unittest.mock import MagicMock

from libsafety_contact import (
    any_robot_hazard_contact,
    find_hazard_object_name,
    hazard_geom_ids,
    robot_geom_ids,
)


def _sim(names, contacts, groups=None):
    sim = MagicMock()
    sim.model.ngeom = len(names)
    sim.model.geom_id2name.side_effect = lambda i: names[i]
    sim.model.geom_group = groups if groups is not None else [0] * len(names)
    sim.data.ncon = len(contacts)
    sim.data.contact = [MagicMock(geom1=a, geom2=b) for a, b in contacts]
    return sim


def test_robot_geom_ids_returns_indices_not_names():
    sim = _sim(["robot0_link4", "table", "gripper0_finger1"], [])
    assert robot_geom_ids(sim) == [0, 2]


def test_robot_geom_ids_skips_unnamed_and_nonvisible_geoms():
    sim = _sim(["robot0_link4", None, "robot0_link5"], [], groups=[0, 0, 1])
    assert robot_geom_ids(sim) == [0]


def test_detects_contact_in_either_slot():
    sim = _sim(["robot0_link4", "hazard_geom"], [(0, 1)])
    assert any_robot_hazard_contact(sim, [1]) is True
    sim = _sim(["robot0_link4", "hazard_geom"], [(1, 0)])
    assert any_robot_hazard_contact(sim, [1]) is True


def test_no_contact_when_hazard_uninvolved():
    sim = _sim(["robot0_link4", "hazard_geom", "table"], [(0, 2)])
    assert any_robot_hazard_contact(sim, [1]) is False


def test_empty_hazard_list_is_false_not_crash():
    sim = _sim(["robot0_link4", "hazard_geom"], [(0, 1)])
    assert any_robot_hazard_contact(sim, []) is False


def test_regression_indices_are_never_compared_against_names():
    """The original defect: integer ids tested for membership in a name list."""
    ids = robot_geom_ids(_sim(["robot0_link4", "gripper0_finger1"], []))
    assert all(isinstance(i, int) for i in ids), "ids must stay in index space"
    assert not any(i in {"robot0_link4", "gripper0_finger1"} for i in ids)


# --- hazard geom resolution (Task 1.2 wiring) ----------------------------

def test_hazard_geom_ids_matches_prefixed_geoms():
    sim = _sim(["robot0_link4", "bottle_of_perfume_1_g0",
                "bottle_of_perfume_1_g1", "table"], [])
    assert hazard_geom_ids(sim, "bottle_of_perfume_1") == [1, 2]


def test_hazard_geom_ids_matches_exact_name():
    sim = _sim(["robot0_link4", "moka_pot_1"], [])
    assert hazard_geom_ids(sim, "moka_pot_1") == [1]


def test_hazard_geom_ids_empty_for_unknown_object():
    sim = _sim(["robot0_link4", "table"], [])
    assert hazard_geom_ids(sim, "nonexistent_1") == []


def test_hazard_geom_ids_empty_for_empty_name():
    """A suite with no resolvable hazard must yield [] rather than everything."""
    sim = _sim(["robot0_link4", "table"], [])
    assert hazard_geom_ids(sim, "") == []
    assert hazard_geom_ids(sim, None) == []


def test_hazard_geom_ids_skips_invisible_geoms():
    sim = _sim(["hazard_1_g0", "hazard_1_g1"], [], groups=[0, 1])
    assert hazard_geom_ids(sim, "hazard_1") == [0]


def test_end_to_end_resolve_then_detect():
    """The full path a driver takes: name -> geom ids -> contact bool."""
    sim = _sim(["robot0_link4", "perfume_1_g0", "table"], [(0, 1)])
    ids = hazard_geom_ids(sim, "perfume_1")
    assert ids == [1]
    assert any_robot_hazard_contact(sim, ids) is True


# --- shared hazard-name resolver -----------------------------------------

def _env_with_constraints(constraints):
    env = MagicMock()
    env.parsed_problem = {"constraints": constraints}
    return env


def test_find_hazard_takes_the_first_checkrobotcontact():
    env = _env_with_constraints([
        ["checkcontact", "bowl_1", "perfume_1"],
        ["checkrobotcontact", "perfume_1"],
        ["checkrobotcontact", "box_1"],
    ])
    assert find_hazard_object_name(env) == "perfume_1"


def test_find_hazard_none_when_no_constraints_block():
    """The whole affordance suite: 0/15 files declare :constraints."""
    assert find_hazard_object_name(_env_with_constraints([])) is None


def test_find_hazard_none_when_only_checkcontact():
    env = _env_with_constraints([["checkcontact", "bowl_1", "perfume_1"]])
    assert find_hazard_object_name(env) is None


def test_find_hazard_falls_back_to_nested_env():
    inner = MagicMock()
    inner.parsed_problem = {"constraints": [["checkrobotcontact", "hazard_1"]]}
    outer = MagicMock()
    outer.parsed_problem = None
    outer.env = inner
    assert find_hazard_object_name(outer) == "hazard_1"


def test_find_hazard_handles_missing_parsed_problem():
    env = MagicMock()
    env.parsed_problem = None
    env.env = None
    assert find_hazard_object_name(env) is None
