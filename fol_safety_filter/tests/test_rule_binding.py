"""Role binding: lifting one scene-independent record over a concrete scene,
and the ordering metadata the effect dispatcher is required to sort on."""

from __future__ import annotations

import pytest

from fol_safety_filter.rule_memory import (
    SceneBindings,
    parse_skill_md,
    resolve,
    resolve_with_unknowns,
)

SPHERE_TEMPLATE = """---
schema_version: 1
id: {rid}
revision: 1
description: Keep {subject} clear of the target.
roles:
  subject: {{binding: {subject}}}
  target: {{binding: {target}}}
applies_when: "NEAR(SUBJECT, TARGET, {radius})"
requires:
  predicate: avoid_sphere
  warning_r: {radius}
  hard_r: 0.08
  push_hard: 0.06
violation_action: avoid
unknown_action: stop_and_report
---

Body, never executed.
"""

SUBJECT_ONLY = """---
schema_version: 1
id: spillable_rotation_lock
revision: 1
description: Cap the wrist rate while carrying something spillable.
roles:
  subject: {binding: currently_grasped}
applies_when: "IS_SPILLABLE(SUBJECT)"
requires:
  predicate: limit_angular_rate
  omega_max: 0.5
violation_action: slow
unknown_action: defer_and_refresh
---

Body.
"""


def sphere_record(rid="eef_obstacle_avoid", subject="eef", target="obstacles",
                  radius="0.20"):
    text = SPHERE_TEMPLATE.format(rid=rid, subject=subject, target=target,
                                  radius=radius)
    return parse_skill_md(text, f"{rid}/SKILL.md")


def checkpoint(name, pos, warning_r=0.15):
    return {"name": name, "pos": pos, "warning_r": warning_r,
            "hard_r": 0.08, "push": 0.06}


# ---- obstacles target ----------------------------------------------------

def test_one_record_over_three_obstacles_yields_three_rules():
    record = sphere_record()
    # Deliberately not alphabetical: `obstacle_names` order is the contract.
    scene = SceneBindings(obstacle_names=["moka_pot", "akita_bowl", "plate"])
    rules = resolve(record, scene)

    assert [r.primary_object for r in rules] == ["moka_pot", "akita_bowl", "plate"]
    assert [r.target_index for r in rules] == [0, 1, 2]
    assert all(r.subject_rank == 0 and r.checkpoint_index == 0 for r in rules)
    assert all(r.record_id == "eef_obstacle_avoid" for r in rules)
    assert [r.name for r in rules] == [
        "eef_obstacle_avoid::eef::moka_pot",
        "eef_obstacle_avoid::eef::akita_bowl",
        "eef_obstacle_avoid::eef::plate",
    ]
    assert all(r.bindings == {"SUBJECT": "eef", "TARGET": r.primary_object}
               for r in rules)


def test_emitted_rule_carries_the_records_effect_and_metadata():
    record = sphere_record()
    rules = resolve(record, SceneBindings(obstacle_names=["moka_pot"]))
    rule = rules[0]

    assert rule.formula == record.applies_when
    assert rule.parsed is not None
    assert rule.cbf_type == "avoid_sphere"
    assert rule.cbf_params == {
        "warning_r": 0.20, "hard_r": 0.08, "push_hard": 0.06,
        "cancel_scale_on_target_corridor": 1.0, "shape": "sphere",
    }
    # A copy, so mutating one rule's params cannot reach the record.
    rule.cbf_params["warning_r"] = 9.0
    assert record.requires.params["warning_r"] == 0.20

    assert rule.violation_action == "avoid"
    assert rule.description == record.description
    assert rule.level == 1
    assert not hasattr(rule, "checkpoint")


def test_empty_obstacle_list_is_a_no_op():
    record = sphere_record()
    assert resolve(record, SceneBindings(obstacle_names=[])) == []


# ---- arm_checkpoints subject --------------------------------------------

def test_arm_checkpoints_over_two_obstacles_yields_the_cross_product():
    record = sphere_record(rid="arm_link_obstacle_avoid",
                           subject="arm_checkpoints", radius="0.15")
    scene = SceneBindings(
        obstacle_names=["moka_pot", "akita_bowl"],
        arm_checkpoints=[checkpoint("wrist", (0.1, 0.0, 0.9)),
                         checkpoint("forearm", (0.0, 0.0, 0.95))],
    )
    rules = resolve(record, scene)

    assert len(rules) == 4
    # Obstacle-major: for each target, the checkpoints in list order.
    assert [(r.target_index, r.checkpoint_index) for r in rules] == [
        (0, 0), (0, 1), (1, 0), (1, 1)
    ]
    assert all(r.subject_rank == 1 for r in rules)
    assert [r.name for r in rules] == [
        "arm_link_obstacle_avoid::wrist::moka_pot",
        "arm_link_obstacle_avoid::forearm::moka_pot",
        "arm_link_obstacle_avoid::wrist::akita_bowl",
        "arm_link_obstacle_avoid::forearm::akita_bowl",
    ]
    assert [r.bindings["SUBJECT"] for r in rules] == [
        "wrist", "forearm", "wrist", "forearm"]
    # The source checkpoint travels with the rule so the dispatcher can read
    # its position without re-deriving the arm geometry.
    assert rules[0].checkpoint is scene.arm_checkpoints[0]
    assert rules[1].checkpoint is scene.arm_checkpoints[1]

    # Emission order already equals the dispatcher's sort key.
    key = lambda r: (r.target_index, r.subject_rank, r.checkpoint_index)
    assert [key(r) for r in rules] == sorted(key(r) for r in rules)


def test_no_arm_checkpoints_is_a_no_op():
    record = sphere_record(subject="arm_checkpoints")
    scene = SceneBindings(obstacle_names=["moka_pot"], arm_checkpoints=[])
    assert resolve(record, scene) == []


def test_eef_and_arm_records_interleave_by_the_sort_key():
    eef = sphere_record(rid="eef_obstacle_avoid", subject="eef")
    arm = sphere_record(rid="arm_link_obstacle_avoid", subject="arm_checkpoints")
    scene = SceneBindings(
        obstacle_names=["moka_pot", "akita_bowl"],
        arm_checkpoints=[checkpoint("wrist", (0.1, 0.0, 0.9))],
    )
    merged = resolve(eef, scene) + resolve(arm, scene)
    merged.sort(key=lambda r: (r.target_index, r.subject_rank, r.checkpoint_index))
    assert [r.name for r in merged] == [
        "eef_obstacle_avoid::eef::moka_pot",
        "arm_link_obstacle_avoid::wrist::moka_pot",
        "eef_obstacle_avoid::eef::akita_bowl",
        "arm_link_obstacle_avoid::wrist::akita_bowl",
    ]


# ---- subject-only records ------------------------------------------------

def test_subject_only_record_yields_one_rule_with_no_primary_object():
    record = parse_skill_md(SUBJECT_ONLY, "spillable_rotation_lock/SKILL.md")
    scene = SceneBindings(obstacle_names=["moka_pot"], grasped="soup_bowl")
    rules = resolve(record, scene)

    assert len(rules) == 1
    rule = rules[0]
    assert rule.primary_object is None
    assert rule.target_index == 0
    assert rule.subject_rank == 0
    assert rule.name == "spillable_rotation_lock::currently_grasped::-"
    assert rule.bindings == {"SUBJECT": "soup_bowl"}
    assert rule.cbf_type == "limit_angular_rate"
    assert rule.cbf_params == {"omega_max": 0.5}


def test_currently_grasped_with_nothing_grasped_is_a_no_op():
    record = parse_skill_md(SUBJECT_ONLY, "spillable_rotation_lock/SKILL.md")
    assert resolve(record, SceneBindings(obstacle_names=["moka_pot"])) == []


# ---- objects_with_fact target -------------------------------------------

def test_objects_with_fact_selects_true_objects_in_sorted_order():
    record = sphere_record(rid="fragile_standoff",
                           target="objects_with_fact:IS_FRAGILE")
    table = {
        "wine_glass": True,
        "akita_bowl": True,
        "steel_block": False,
        "mystery_jar": None,
    }
    scene = SceneBindings(
        obstacle_names=["steel_block", "wine_glass", "mystery_jar", "akita_bowl"],
        facts=lambda obj, fact: table.get(obj) if fact == "IS_FRAGILE" else None,
    )
    rules, unknowns = resolve_with_unknowns(record, scene)

    # Sorted name order, not the obstacle order.
    assert [r.primary_object for r in rules] == ["akita_bowl", "wine_glass"]
    assert [r.target_index for r in rules] == [0, 1]
    # An unknown fact is neither true nor false: not bound, but reported.
    assert unknowns == ["mystery_jar"]
    assert resolve(record, scene) == rules


def test_objects_with_fact_defaults_to_obstacles_plus_the_payload():
    record = sphere_record(rid="fragile_standoff",
                           target="objects_with_fact:IS_FRAGILE")
    scene = SceneBindings(obstacle_names=["steel_block"], grasped="wine_glass",
                          facts=lambda obj, fact: obj == "wine_glass")
    assert [r.primary_object for r in resolve(record, scene)] == ["wine_glass"]


def test_objects_with_fact_honours_an_explicit_object_universe():
    record = sphere_record(rid="fragile_standoff",
                           target="objects_with_fact:IS_FRAGILE")
    scene = SceneBindings(obstacle_names=["steel_block"],
                          object_names=["ramekin", "steel_block"],
                          facts=lambda obj, fact: obj == "ramekin")
    assert [r.primary_object for r in resolve(record, scene)] == ["ramekin"]


def test_objects_with_fact_with_no_true_object_is_a_no_op():
    record = sphere_record(rid="fragile_standoff",
                           target="objects_with_fact:IS_FRAGILE")
    scene = SceneBindings(obstacle_names=["steel_block", "mystery_jar"],
                          facts=lambda obj, fact: None)
    rules, unknowns = resolve_with_unknowns(record, scene)
    assert rules == []
    assert unknowns == ["mystery_jar", "steel_block"]


def test_default_fact_oracle_knows_nothing():
    record = sphere_record(rid="fragile_standoff",
                           target="objects_with_fact:IS_FRAGILE")
    rules, unknowns = resolve_with_unknowns(
        record, SceneBindings(obstacle_names=["plate"]))
    assert rules == []
    assert unknowns == ["plate"]
