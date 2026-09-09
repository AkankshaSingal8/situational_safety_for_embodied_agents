"""S4: the rotation branch reads `omega_max`, so producers must write that key.

`cbf_mapper.py:184` reads `params.get("omega_max", 0.25)`. The predicate
templates in `primitives.py:356,407` correctly write `omega_max` -- but three
producers wrote `angular_limit` instead (`filter.py:466,547`,
`rule_composer.py:76`), which no consumer reads, so every rule from those paths
silently fell back to the 0.25 rad/s default instead of its intended
0.15 / 0.15 / 0.10.

Also pinned: `tilt_max` must FOLD with min() across multiple rotation rules
rather than being overwritten, matching the adjacent v_limit / omega_limit folds.
"""
import math

import numpy as np

from fol_safety_filter.cbf_mapper import CBFMapper
from fol_safety_filter.kb import ActiveConstraint, FOLRule, parse_formula


def _rot(params):
    rule = FOLRule(
        name="test_rot",
        formula="TILTED(eef)",
        parsed=parse_formula("TILTED(eef)"),
        bindings={},
        cbf_type="rotation",
        cbf_params=params,
        violation_action="slow",
    )
    return ActiveConstraint(
        rule=rule, primary_object=None, obj_pos=None, obj_quat=None, obj_bbox=None
    )


IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def test_omega_max_is_honoured():
    out = CBFMapper().map([_rot({"omega_max": 0.15})], IDENTITY_QUAT)
    assert out.angular_limit == 0.15


def test_default_applies_only_when_no_key_given():
    out = CBFMapper().map([_rot({})], IDENTITY_QUAT)
    assert out.angular_limit == 0.25


def test_tightest_omega_wins_across_rules():
    out = CBFMapper().map(
        [_rot({"omega_max": 0.20}), _rot({"omega_max": 0.10})], IDENTITY_QUAT
    )
    assert out.angular_limit == 0.10, "omega_limit must fold with min()"


def test_tightest_tilt_wins_across_rules():
    # Tight rule FIRST so that "last one wins" and "min" give different answers:
    # overwrite -> 20 deg, correct fold -> 8 deg.
    out = CBFMapper().map(
        [_rot({"tilt_max_deg": 8.0}), _rot({"tilt_max_deg": 20.0})], IDENTITY_QUAT
    )
    assert out.tilt_max_rad == math.radians(8.0), (
        "tilt_max must fold with min(), not be overwritten by the last rule"
    )


def test_tightest_omega_wins_regardless_of_order():
    a = CBFMapper().map([_rot({"omega_max": 0.10}), _rot({"omega_max": 0.20})], IDENTITY_QUAT)
    b = CBFMapper().map([_rot({"omega_max": 0.20}), _rot({"omega_max": 0.10})], IDENTITY_QUAT)
    assert a.angular_limit == b.angular_limit == 0.10


def test_no_producer_still_writes_the_unread_angular_limit_key():
    """Regression guard: `angular_limit` in cbf_params is read by nothing."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "fol_safety_filter"
    offenders = []
    for path in root.glob("*.py"):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            # cbf_params dicts carrying angular_limit -- not MappedCBFSet's own
            # angular_limit attribute, which is the mapper's output field.
            if re.search(r'["\']angular_limit["\']\s*:', line):
                offenders.append(f"{path.name}:{i}")
    assert not offenders, f"cbf_params must use omega_max: {offenders}"
