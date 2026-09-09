"""`target_exclusions` must actually reach the binding resolver.

`_add_semantic_rules` has always been called with
`target_exclusions=set(self._obstacle_names)` (filter.py:389-390), but the value
was never forwarded to `_resolve_bindings`, so objects already governed by the
geometric layer could also be bound as semantic-rule subjects and constrained
twice by two mechanisms.

This became reachable once S5 was fixed: `_props_from_name` marks pot/bowl/milk
spillable, so SPILL_RISK would bind A to the obstacle itself.
"""
import numpy as np

from fol_safety_filter.filter import FOLSafetyFilter
from fol_safety_filter.primitives import ObjectState


def _obj(name, **props):
    return ObjectState(
        name=name,
        pos=np.zeros(3),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        bbox_half=np.full(3, 0.05),
        **props,
    )


SPILL_SEED = {"name": "SPILL_RISK", "formula": "IS_SPILLABLE(A) AND HOLDING(eef, A)"}


def test_excluded_object_is_not_bound():
    f = FOLSafetyFilter()
    f._object_states = {
        "moka_pot_obstacle_1": _obj("moka_pot_obstacle_1", is_spillable=True),
        "akita_black_bowl_1": _obj("akita_black_bowl_1", is_spillable=True),
    }
    bindings = f._resolve_bindings(SPILL_SEED, {"moka_pot_obstacle_1"})
    assert bindings is not None
    assert bindings["A"] == "akita_black_bowl_1", (
        "the excluded obstacle must not be bound as a semantic-rule subject"
    )


def test_returns_none_when_every_candidate_is_excluded():
    f = FOLSafetyFilter()
    f._object_states = {"moka_pot_obstacle_1": _obj("moka_pot_obstacle_1", is_spillable=True)}
    assert f._resolve_bindings(SPILL_SEED, {"moka_pot_obstacle_1"}) is None


def test_no_exclusions_binds_normally():
    f = FOLSafetyFilter()
    f._object_states = {"moka_pot_1": _obj("moka_pot_1", is_spillable=True)}
    assert f._resolve_bindings(SPILL_SEED, None)["A"] == "moka_pot_1"


def test_regression_exclusions_are_not_silently_dropped():
    """Before the fix, passing exclusions changed nothing at all."""
    f = FOLSafetyFilter()
    f._object_states = {"obst": _obj("obst", is_spillable=True)}
    with_excl = f._resolve_bindings(SPILL_SEED, {"obst"})
    without = f._resolve_bindings(SPILL_SEED, None)
    assert with_excl != without, "exclusions must change the outcome"
