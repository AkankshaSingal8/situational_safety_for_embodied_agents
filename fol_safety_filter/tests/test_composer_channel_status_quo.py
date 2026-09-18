"""Pin the measured status quo of the pre-memory rule path.

Two defects in the composer path were found while deciding which rule records
could safely be default-loaded.  Neither is fixed here: fixing either one
changes the filter's behaviour, and every number in
``fol_safety_filter/README.md`` was produced with both present.  They are
pinned instead, so that a later fix has to be a deliberate, measured change
rather than an accident that silently moves the results.

Defect 1 -- unreachable parameter.  ``rule_composer.py:76`` and
``filter.py:427,508`` author rotation rules as ``{"angular_limit": ...}``,
while ``cbf_mapper.py:184`` reads ``params.get("omega_max", 0.25)``.  The
authored value is dead and every rotation lock runs at 0.25 rad/s.

Defect 2 -- self-falsifying rule.  ``vlm_grounder._props_from_name`` keys
``IS_SPILLABLE`` on pot/bottle/cup/bowl/mug/milk, but
``VLMGrounder._heuristic_ground`` sets ``is_open`` only for
bowl/cup/glass/mug.  For a pot or a bottle the composer therefore authors a
rotation rule whose own precondition can never hold.
"""

from __future__ import annotations

import numpy as np
import pytest

from fol_safety_filter.cbf_mapper import CBFMapper
from fol_safety_filter.filter import VLMGrounder
from fol_safety_filter.kb import ActiveConstraint, FOLRule, parse_formula
from fol_safety_filter.primitives import IS_SPILLABLE, ObjectState, RobotState
from fol_safety_filter.rule_composer import compose_rules
from fol_safety_filter.vlm_grounder import _props_from_name


def _state(objects):
    return RobotState(
        ee_pos=np.zeros(3),
        ee_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        ee_vel=np.zeros(3),
        joint_pos=np.zeros(7),
        joint_vel=np.zeros(7),
        joint_limits_lo=np.full(7, -2.8),
        joint_limits_hi=np.full(7, 2.8),
        gripper_width=0.02,
        objects={o.name: o for o in objects},
    )


def _grounded(name: str) -> ObjectState:
    obj = ObjectState(
        name=name,
        pos=np.zeros(3),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        bbox_half=np.full(3, 0.04),
    )
    VLMGrounder(use_vlm=False)._heuristic_ground(name, obj)
    return obj


# --------------------------------------------------------------------------- #
# Defect 1: the authored angular limit is unreachable
# --------------------------------------------------------------------------- #

def test_authored_angular_limit_is_ignored_by_the_mapper():
    """A rotation rule authored with angular_limit=0.1 still runs at 0.25."""
    formula = "IS_SPILLABLE(cup)"
    rule = FOLRule(
        name="SPILLABLE_LOCK_CUP",
        formula=formula,
        parsed=parse_formula(formula),
        bindings={},
        cbf_type="rotation",
        cbf_params={"angular_limit": 0.1},   # as rule_composer.py:76 writes it
        violation_action="slow",
        primary_object="cup",
        level=2,
    )
    mapped = CBFMapper().map(
        [ActiveConstraint(rule=rule, primary_object="cup", obj_pos=None,
                          obj_quat=None, obj_bbox=None)],
        ee_quat=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    assert mapped.angular_limit == pytest.approx(0.25), (
        "cbf_mapper.py:184 reads 'omega_max'; if this now equals 0.1 the "
        "key-name defect has been fixed and the rotation channel has been "
        "tightened -- re-measure before accepting it"
    )


def test_omega_max_is_the_key_that_actually_works():
    """Spelled omega_max, the authored value does take effect."""
    formula = "IS_SPILLABLE(cup)"
    rule = FOLRule(
        name="SPILLABLE_LOCK_CUP",
        formula=formula,
        parsed=parse_formula(formula),
        bindings={},
        cbf_type="rotation",
        cbf_params={"omega_max": 0.1},
        violation_action="slow",
        primary_object="cup",
        level=2,
    )
    mapped = CBFMapper().map(
        [ActiveConstraint(rule=rule, primary_object="cup", obj_pos=None,
                          obj_quat=None, obj_bbox=None)],
        ee_quat=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    assert mapped.angular_limit == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# Defect 2: for the reported obstacles the composed rotation rule cannot fire
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "name, expect_prop, expect_fires",
    [
        # keyword lists disagree -> rule authored, precondition never holds
        ("moka_pot_obstacle", True, False),
        ("wine_bottle_obstacle", True, False),
        # keyword lists agree -> rule authored and able to fire
        ("akita_black_bowl_1", True, True),
        ("white_yellow_mug_1", True, True),
    ],
)
def test_composed_rotation_rule_reachability(name, expect_prop, expect_fires):
    props, radius = _props_from_name(name)
    assert ("IS_SPILLABLE" in props) is expect_prop

    composed = compose_rules(
        {
            "obstacle_obs_keys": [name],
            "obstacle_radii": {name: radius},
            "obstacle_properties": {name: props},
        }
    )
    lock = [r for r in composed if r.cbf_type == "rotation"]
    assert lock, f"composer did not author a rotation rule for {name}"

    state = _state([_grounded(name)])
    assert IS_SPILLABLE(state, name) is expect_fires
    assert lock[0].parsed.eval(state, lock[0].bindings) is expect_fires


def test_safelibero_obstacles_leave_the_rotation_channel_inert():
    """End to end on the reported obstacle: a lock is authored, none engages."""
    name = "moka_pot_obstacle"
    props, radius = _props_from_name(name)
    composed = compose_rules(
        {
            "obstacle_obs_keys": [name],
            "obstacle_radii": {name: radius},
            "obstacle_properties": {name: props},
        }
    )
    state = _state([_grounded(name)])
    active = [
        ActiveConstraint(rule=r, primary_object=r.primary_object,
                         obj_pos=None, obj_quat=None, obj_bbox=None)
        for r in composed
        if r.parsed.eval(state, r.bindings)
    ]
    mapped = CBFMapper().map(active, ee_quat=np.array([0.0, 0.0, 0.0, 1.0]))
    assert mapped.angular_limit is None
    assert mapped.pose_constrained is False
