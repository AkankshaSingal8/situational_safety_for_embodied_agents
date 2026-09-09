"""Tests for the independent robot-hazard contact scorer.

The benchmark's own CheckRobotContact (bddl_base_domain.py:1057) appends geom
INDICES to its group while _check_contact resolves contacts to NAMES and tests
membership, so it can never return True. These tests pin the corrected
behaviour: work in index space throughout.
"""
from unittest.mock import MagicMock

from libsafety_contact import robot_geom_ids, any_robot_hazard_contact


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
