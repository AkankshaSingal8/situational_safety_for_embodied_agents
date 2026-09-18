"""The records must be load-bearing, not decorative.

Before the reroute, the knowledge base could not change what ran: `_apply_cbf`
looped over `self._obstacle_names` with literal geometry, so deleting a rule
stopped it being *counted* and nothing else. A file format layered over that
loop would look like a decomposition and change nothing.

These tests are what distinguishes the two. Each asserts in both directions —
that a record present produces a correction, and that the same record absent
produces none — because only the pair rules out the geometry having quietly
stayed in Python.

The sharpest of them is `test_editing_the_arm_radius_changes_the_action`:
deleting a record could pass while the numbers inside it were still ignored.
Editing one cannot.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from fol_safety_filter.cbf_mapper import MappedCBFSet
from fol_safety_filter.filter import SAFETY_MEMORY_ROOT, FOLSafetyFilter
from fol_safety_filter.primitives import ObjectState, RobotState
from fol_safety_filter.rule_memory.testing import install_memory

OBSTACLE = "moka_pot_obstacle"
OBS_POS = np.array([0.0, 0.0, 0.85])


def _filter():
    f = FOLSafetyFilter.__new__(FOLSafetyFilter)
    f._obstacle_names = [OBSTACLE]
    f._obstacle_radii = {}
    f._obstacle_ellipsoids = {}
    f._vision_targets_xy = []
    f._vision_target_xy = None
    f._vision_fallback_active = False
    f._fallback_vlim = 0.010
    f.velocity_limit_default = None
    f._target_name = None
    return f


def _state(ee_pos):
    return RobotState(
        ee_pos=np.asarray(ee_pos, dtype=np.float64),
        ee_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        ee_vel=np.zeros(3),
        joint_pos=np.zeros(7),
        joint_vel=np.zeros(7),
        joint_limits_lo=np.full(7, -2.8),
        joint_limits_hi=np.full(7, 2.8),
        gripper_width=0.02,
        objects={
            OBSTACLE: ObjectState(
                name=OBSTACLE,
                pos=OBS_POS.copy(),
                quat=np.array([0.0, 0.0, 0.0, 1.0]),
                bbox_half=np.full(3, 0.04),
            )
        },
    )


def _cbf_set():
    return MappedCBFSet(semantic_envelopes=[], collision_envelopes=[])


def _checkpoint(distance: float, name: str = "robot0_link6"):
    """A checkpoint `distance` metres from the obstacle, carrying NO radii.

    No radii is the point: the zone then comes from the record, so editing the
    record is what decides whether this checkpoint is inside it.
    """
    return {"pos": OBS_POS + np.array([distance, 0.0, 0.0]), "name": name}


def _run(f, ee_pos, u_cmd, checkpoints=()):
    state = _state(ee_pos)
    u_safe, _ = f._apply_cbf(
        np.asarray(u_cmd, dtype=np.float64), state, _cbf_set(), list(checkpoints)
    )
    return u_safe


# --------------------------------------------------------------------------- #
# Deleting a record removes exactly one channel
# --------------------------------------------------------------------------- #

# EEF outside every zone, one arm checkpoint 0.10 m from the obstacle: inside
# the record's 0.15 m warning radius, outside its 0.08 m hard radius.
FAR_EEF = OBS_POS + np.array([0.90, 0.40, 0.20])
INWARD = [-0.02, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_arm_record_present_corrects_the_action():
    f = _filter()
    install_memory(f, [OBSTACLE], arm_checkpoints=[_checkpoint(0.10)])
    u_safe = _run(f, FAR_EEF, INWARD, [_checkpoint(0.10)])
    assert not np.allclose(u_safe[:3], INWARD[:3]), (
        "the arm channel did not act: with the record enabled and a "
        "checkpoint inside its warning radius, u_safe must differ from u_cmd"
    )


def test_arm_record_absent_leaves_the_action_untouched():
    f = _filter()
    install_memory(
        f, [OBSTACLE],
        arm_checkpoints=[_checkpoint(0.10)],
        enabled_ids=["eef_obstacle_avoid"],   # arm record withheld
    )
    u_safe = _run(f, FAR_EEF, INWARD, [_checkpoint(0.10)])
    assert np.allclose(u_safe[:3], INWARD[:3]), (
        "removing arm_link_obstacle_avoid did not remove the arm channel, so "
        "its geometry is still coming from somewhere other than the record"
    )


def test_arm_record_absent_also_disappears_from_the_fired_set():
    f = _filter()
    install_memory(f, [OBSTACLE], arm_checkpoints=[_checkpoint(0.10)])
    state = _state(FAR_EEF)
    from fol_safety_filter.rule_memory import register_checkpoint_pseudo_objects

    register_checkpoint_pseudo_objects(state, [_checkpoint(0.10)])
    with_record = {ac.rule.record_id for ac in f.kb.evaluate(state)}
    assert "arm_link_obstacle_avoid" in with_record

    g = _filter()
    install_memory(
        g, [OBSTACLE],
        arm_checkpoints=[_checkpoint(0.10)],
        enabled_ids=["eef_obstacle_avoid"],
    )
    without = {ac.rule.record_id for ac in g.kb.evaluate(state)}
    assert "arm_link_obstacle_avoid" not in without


def test_eef_record_present_and_absent():
    """The same pair for the end-effector channel, arm checkpoints withheld."""
    near_eef = OBS_POS + np.array([0.12, 0.0, 0.0])

    f = _filter()
    install_memory(f, [OBSTACLE])
    corrected = _run(f, near_eef, INWARD)
    assert not np.allclose(corrected[:3], INWARD[:3])

    g = _filter()
    install_memory(g, [OBSTACLE], enabled_ids=["arm_link_obstacle_avoid"])
    untouched = _run(g, near_eef, INWARD)
    assert np.allclose(untouched[:3], INWARD[:3]), (
        "removing eef_obstacle_avoid did not remove the end-effector channel"
    )


def test_empty_memory_applies_no_correction_at_all():
    """The strongest form: no records, no geometry, anywhere."""
    f = _filter()
    install_memory(f, [OBSTACLE], enabled_ids=[])
    inside_hard = OBS_POS + np.array([0.03, 0.0, 0.0])
    u_safe = _run(f, inside_hard, INWARD)
    assert np.allclose(u_safe[:3], INWARD[:3]), (
        "a correction was applied with no rule records loaded, so geometry "
        "remains hardcoded somewhere in the filter"
    )


# --------------------------------------------------------------------------- #
# Editing a record's numbers changes the action
# --------------------------------------------------------------------------- #

def _store_with_arm_radius(tmp_path, warning_r: str):
    """Copy the real store, rewriting only arm_link_obstacle_avoid's radius."""
    root = tmp_path / "safety_memory"
    shutil.copytree(SAFETY_MEMORY_ROOT, root)
    skill = root / "arm_link_obstacle_avoid" / "SKILL.md"
    text = skill.read_text()
    assert "  warning_r: 0.15\n" in text
    skill.write_text(text.replace("  warning_r: 0.15\n",
                                  f"  warning_r: {warning_r}\n", 1))
    return root


@pytest.mark.parametrize(
    "warning_r, distance, should_act",
    [
        # 0.20 m away: outside the authored 0.15 radius, inside a widened 0.25
        ("0.15", 0.20, False),
        ("0.25", 0.20, True),
    ],
)
def test_editing_the_arm_radius_changes_the_action(
    tmp_path, warning_r, distance, should_act
):
    """Deleting a record could pass while its numbers were ignored; this cannot.

    Same scene, same checkpoint, same commanded action — only the number in
    the record differs. If the parameters were not load-bearing, both rows
    would agree.
    """
    from fol_safety_filter.kb import FOLKnowledgeBase
    from fol_safety_filter.rule_memory import SceneBindings, install_rules

    checkpoints = [_checkpoint(distance)]
    f = _filter()
    f.kb = FOLKnowledgeBase()
    install_rules(
        f.kb,
        SceneBindings(
            obstacle_names=[OBSTACLE],
            arm_checkpoints=checkpoints,
            grasped=None,
            facts=lambda _o, _f: None,
            object_names=[OBSTACLE],
        ),
        root=_store_with_arm_radius(tmp_path, warning_r),
    )

    u_safe = _run(f, FAR_EEF, INWARD, checkpoints)
    acted = not np.allclose(u_safe[:3], INWARD[:3])
    assert acted is should_act, (
        f"warning_r={warning_r} at {distance} m: expected "
        f"{'a correction' if should_act else 'no correction'}, got the "
        f"opposite — the record's radii are not reaching the projection"
    )
