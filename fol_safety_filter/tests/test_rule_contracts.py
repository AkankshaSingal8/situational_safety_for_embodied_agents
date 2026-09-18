"""Execute every rule record's own `tests.yaml` against the live memory.

A record's activation contract lives beside the record, so that adding a rule
means adding its positive, negative and boundary cases in the same directory
rather than remembering to extend a central test file. This module is the
runner: it loads the real `fol_safety_filter/safety_memory/` store, lifts each
record over the scene a case describes, and evaluates the resulting rules.

Each case therefore exercises binding *and* the formula. A case that expects
`expect_resolved: 0` is asserting that nothing bound (the record had no target
in this scene); a case that expects `silent` with a nonzero `expect_resolved`
is asserting that something bound and correctly declined to fire. Those are
different failures and the fixtures distinguish them deliberately.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pytest
import yaml

from fol_safety_filter.primitives import ObjectState, RobotState
from fol_safety_filter.rule_memory import (
    SceneBindings,
    load_library,
    register_checkpoint_pseudo_objects,
    resolve_with_unknowns,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
MEMORY_ROOT = os.path.join(os.path.dirname(_HERE), "safety_memory")


def _library():
    # every manifested record, enabled or not: a contract applies to a record,
    # not to whether the manifest currently switches it on.
    lib = load_library(MEMORY_ROOT)
    return {r.id: r for r in lib.all_records}


def _load_cases():
    cases = []
    for rule_id in sorted(_library()):
        path = os.path.join(MEMORY_ROOT, rule_id, "tests.yaml")
        if not os.path.exists(path):
            cases.append(pytest.param(rule_id, None, id=f"{rule_id}-MISSING"))
            continue
        doc = yaml.safe_load(open(path))
        for case in doc["cases"]:
            cases.append(
                pytest.param(rule_id, case, id=f"{rule_id}-{case['name']}")
            )
    return cases


def _build(scene: Dict):
    """Turn a fixture scene into (SceneBindings, RobotState)."""
    objects: Dict[str, ObjectState] = {}
    facts: Dict[str, Dict[str, Optional[bool]]] = {}

    for name, spec in (scene.get("objects") or {}).items():
        obj = ObjectState(
            name=name,
            pos=np.asarray(spec["pos"], dtype=np.float64),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            bbox_half=np.full(3, 0.04),
        )
        case_facts = spec.get("facts") or {}
        facts[name] = case_facts
        # Mirror the fixture's facts onto the ObjectState fields the IS_*
        # predicates actually read (is_fragile, is_lit, ...).
        for fact, value in case_facts.items():
            setattr(obj, fact.lower(), value)
        objects[name] = obj

    def fact_lookup(obj_name: str, fact: str) -> Optional[bool]:
        return facts.get(obj_name, {}).get(fact)

    checkpoints: List[Dict] = []
    for cp in scene.get("arm_checkpoints") or []:
        checkpoints.append({**cp, "pos": np.asarray(cp["pos"], dtype=np.float64)})

    bindings = SceneBindings(
        obstacle_names=list(scene.get("obstacles") or []),
        arm_checkpoints=checkpoints,
        grasped=scene.get("grasped"),
        facts=fact_lookup,
        object_names=sorted(objects),
    )
    state = RobotState(
        ee_pos=np.asarray(scene["eef_pos"], dtype=np.float64),
        ee_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        ee_vel=np.zeros(3),
        joint_pos=np.zeros(7),
        joint_vel=np.zeros(7),
        joint_limits_lo=np.full(7, -2.8),
        joint_limits_hi=np.full(7, 2.8),
        gripper_width=0.0 if scene.get("grasped") else 0.08,
        objects=objects,
    )
    register_checkpoint_pseudo_objects(state, checkpoints)
    return bindings, state


@pytest.mark.parametrize("rule_id, case", _load_cases())
def test_record_contract(rule_id, case):
    assert case is not None, f"{rule_id}/tests.yaml is missing"

    record = _library()[rule_id]
    bindings, state = _build(case["scene"])
    rules, unknowns = resolve_with_unknowns(record, bindings)

    if "expect_resolved" in case:
        assert len(rules) == case["expect_resolved"], (
            f"{rule_id}/{case['name']}: expected {case['expect_resolved']} "
            f"lifted rule(s), got {len(rules)}: {[r.name for r in rules]}"
        )
    if "expect_unknown" in case:
        assert sorted(unknowns) == sorted(case["expect_unknown"])

    fired = [r for r in rules if r.parsed.eval(state, r.bindings)]

    if case["expect"] == "fires":
        assert fired, (
            f"{rule_id}/{case['name']}: expected activation, none fired "
            f"(lifted {len(rules)}: {[r.name for r in rules]})"
        )
    elif case["expect"] == "silent":
        assert not fired, (
            f"{rule_id}/{case['name']}: expected silence, these fired: "
            f"{[r.name for r in fired]}"
        )
    else:
        pytest.fail(f"unknown expect value {case['expect']!r}")

    if "expect_fired" in case:
        assert len(fired) == case["expect_fired"], (
            f"{rule_id}/{case['name']}: expected {case['expect_fired']} "
            f"firing rule(s), got {len(fired)}: {[r.name for r in fired]}"
        )


def test_every_manifested_record_has_a_contract():
    """A record with no activation cases is a rule nobody has checked."""
    for rule_id in _library():
        path = os.path.join(MEMORY_ROOT, rule_id, "tests.yaml")
        assert os.path.exists(path), f"{rule_id} has no tests.yaml"
        doc = yaml.safe_load(open(path))
        expects = {c["expect"] for c in doc["cases"]}
        assert "fires" in expects, f"{rule_id} has no positive case"
        assert "silent" in expects, f"{rule_id} has no negative case"


def test_every_manifested_record_has_metadata():
    import json

    for rule_id in _library():
        path = os.path.join(MEMORY_ROOT, rule_id, "metadata.json")
        assert os.path.exists(path), f"{rule_id} has no metadata.json"
        doc = json.load(open(path))
        assert doc["id"] == rule_id
        assert doc["provenance"], f"{rule_id} metadata states no provenance"
        assert "evidence" in doc, f"{rule_id} metadata states no evidence"


def test_manifest_holds_exactly_the_five_records():
    """Deliberate tripwire: adding a record should be a conscious act.

    If you added a sixth rule, this failure is expected. Add its id to the
    list below and say in the commit message what the record is for and why
    it is (or is not) default-loaded. If you made it default-loaded, the
    parity check in `test_parity.py` has to still pass, and if it changes rule
    *selection* rather than the projection, parity does not cover it — see
    the design's section 6.1.
    """
    lib = load_library(MEMORY_ROOT)
    assert [r.id for r in lib.all_records] == [
        "arm_link_obstacle_avoid",
        "eef_obstacle_avoid",
        "fragile_speed_limit",
        "overhead_exclusion",
        "spillable_rotation_lock",
    ]
    # Only the two parity-covered spatial records load by default.
    assert lib.enabled_ids() == (
        "arm_link_obstacle_avoid", "eef_obstacle_avoid"
    ), (
        "the default-loaded set changed. Every reported FOL number was "
        "produced by these two records' geometry; enabling a third changes "
        "rule selection, which the golden parity trace does not cover."
    )
