"""Version-1 rule-record contract: one positive case, and one negative test per
malformed class the design's §3.2 says must fail the whole snapshot."""

from __future__ import annotations

import hashlib

import pytest

from fol_safety_filter.rule_memory import schema
from fol_safety_filter.rule_memory.schema import (
    InvalidRuleError,
    load_record,
    parse_skill_md,
)

FRONT_MATTER = """schema_version: 1
id: eef_obstacle_avoid
revision: 3
description: Keep the end-effector out of each obstacle's exclusion zone.
roles:
  subject: {binding: eef}
  target: {binding: obstacles}
applies_when: "NEAR(SUBJECT, TARGET, 0.20)"
requires:
  predicate: avoid_sphere
  warning_r: 0.20
  hard_r: 0.10
  push_hard: 0.08
violation_action: avoid
unknown_action: stop_and_report
"""

BODY = """
# Why

Geometric proximity only. Proves nothing about spillage or grasp security.
"""


def make_text(front_matter: str = FRONT_MATTER, body: str = BODY) -> str:
    return f"---\n{front_matter}---\n{body}"


def write(tmp_path, front_matter: str = FRONT_MATTER, body: str = BODY):
    path = tmp_path / "SKILL.md"
    path.write_text(make_text(front_matter, body), encoding="utf-8")
    return path


def replace(line_start: str, new_line: str) -> str:
    """Swap the front-matter line beginning with ``line_start``."""
    out = []
    hit = False
    for line in FRONT_MATTER.splitlines(keepends=True):
        if line.startswith(line_start):
            hit = True
            if new_line is not None:
                out.append(new_line if new_line.endswith("\n") else new_line + "\n")
        else:
            out.append(line)
    assert hit, f"no front-matter line starts with {line_start!r}"
    return "".join(out)


# ---- positive ------------------------------------------------------------

def test_valid_record_parses_with_hashes_and_defaults(tmp_path):
    record = load_record(write(tmp_path))

    assert record.id == "eef_obstacle_avoid"
    assert record.revision == 3
    assert record.roles == {
        "subject": {"binding": "eef"},
        "target": {"binding": "obstacles"},
    }
    assert record.applies_when == "NEAR(SUBJECT, TARGET, 0.20)"
    assert record.violation_action == "avoid"
    assert record.unknown_action == "stop_and_report"
    assert record.has_target and record.subject_binding == "eef"

    effect = record.requires
    assert effect.predicate == "avoid_sphere"
    # Required params kept, optional params defaulted in.
    assert effect.params == {
        "warning_r": 0.20,
        "hard_r": 0.10,
        "push_hard": 0.08,
        "cancel_scale_on_target_corridor": 1.0,
        "shape": "sphere",
    }

    # Hashes are over the exact bytes between / after the delimiters. The
    # expected values are built here independently rather than via the module's
    # own splitter, so the convention cannot drift silently.
    assert record.front_matter_hash == hashlib.sha256(
        FRONT_MATTER.encode("utf-8")
    ).hexdigest()
    assert record.body_hash == hashlib.sha256(BODY.encode("utf-8")).hexdigest()
    assert record.front_matter_hash != record.body_hash


def test_optional_params_are_honoured_when_present(tmp_path):
    fm = FRONT_MATTER.replace(
        "  push_hard: 0.08\n",
        "  push_hard: 0.08\n  shape: ellipsoid_if_available\n"
        "  cancel_scale_on_target_corridor: 0.6\n",
    )
    record = load_record(write(tmp_path, fm))
    assert record.requires.params["shape"] == "ellipsoid_if_available"
    assert record.requires.params["cancel_scale_on_target_corridor"] == 0.6


def test_subject_only_record_needs_no_target(tmp_path):
    fm = (
        "schema_version: 1\n"
        "id: spillable_rotation_lock\n"
        "revision: 1\n"
        "description: Cap the wrist rate while carrying something spillable.\n"
        "roles:\n"
        "  subject: {binding: currently_grasped}\n"
        'applies_when: "IS_SPILLABLE(SUBJECT)"\n'
        "requires:\n"
        "  predicate: limit_angular_rate\n"
        "  omega_max: 0.5\n"
        "violation_action: slow\n"
        "unknown_action: defer_and_refresh\n"
    )
    record = load_record(write(tmp_path, fm))
    assert not record.has_target
    assert record.target_binding is None
    assert record.requires.params == {"omega_max": 0.5}


def test_body_is_never_interpreted(tmp_path):
    body = "\nrequires:\n  predicate: limit_speed\n  v_max: 99.0\n"
    record = load_record(write(tmp_path, body=body))
    assert record.requires.predicate == "avoid_sphere"


def test_validate_accepts_a_parsed_record(tmp_path):
    record = load_record(write(tmp_path))
    assert schema.validate(record) is record


# ---- structure / front matter -------------------------------------------

def test_missing_front_matter(tmp_path):
    path = tmp_path / "SKILL.md"
    path.write_text("# just markdown\n", encoding="utf-8")
    with pytest.raises(InvalidRuleError, match="front_matter"):
        load_record(path)


def test_unterminated_front_matter(tmp_path):
    path = tmp_path / "SKILL.md"
    path.write_text(f"---\n{FRONT_MATTER}", encoding="utf-8")
    with pytest.raises(InvalidRuleError, match="front_matter"):
        load_record(path)


def test_two_front_matter_blocks(tmp_path):
    path = tmp_path / "SKILL.md"
    path.write_text(f"---\n{FRONT_MATTER}---\n---\nid: other\n---\n", encoding="utf-8")
    with pytest.raises(InvalidRuleError) as exc:
        load_record(path)
    assert "more than one front-matter block" in str(exc.value)


def test_horizontal_rule_deeper_in_the_body_is_allowed(tmp_path):
    body = "\n# Why\n\nGeometric only.\n\n---\n\n## Limitations\n\nNone claimed.\n"
    record = load_record(write(tmp_path, body=body))
    assert record.id == "eef_obstacle_avoid"


def test_non_mapping_front_matter(tmp_path):
    with pytest.raises(InvalidRuleError, match="front_matter"):
        parse_skill_md("---\n- a\n- b\n---\nbody\n", "rec.md")


def test_oversize_file(tmp_path):
    body = BODY + "x" * schema.MAX_RECORD_BYTES
    path = tmp_path / "SKILL.md"
    path.write_text(make_text(body=body), encoding="utf-8")
    with pytest.raises(InvalidRuleError) as exc:
        load_record(path)
    assert "file" in str(exc.value) and "exceeds" in str(exc.value)


@pytest.mark.parametrize("sigil,label", [
    ("  anchored: &a 0.2\n", "anchor"),
    ("  aliased: *a\n", "alias"),
    ("  tagged: !!float 0.2\n", "tag"),
])
def test_yaml_anchors_aliases_and_tags(tmp_path, sigil, label):
    fm = FRONT_MATTER + sigil
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert label in str(exc.value)


def test_error_message_names_the_file(tmp_path):
    path = write(tmp_path, replace("id:", "id: Bad-Id"))
    with pytest.raises(InvalidRuleError) as exc:
        load_record(path)
    assert str(path) in str(exc.value)


# ---- top-level keys ------------------------------------------------------

def test_unknown_top_level_key(tmp_path):
    fm = FRONT_MATTER + "priority: 7\n"
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "unknown key" in str(exc.value) and "priority" in str(exc.value)


@pytest.mark.parametrize("key", [
    "schema_version", "id", "revision", "description",
    "applies_when", "violation_action", "unknown_action",
])
def test_missing_required_scalar_key(tmp_path, key):
    fm = replace(f"{key}:", None)
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert key in str(exc.value)


@pytest.mark.parametrize("key,block", [
    ("roles", "roles:\n  subject: {binding: eef}\n  target: {binding: obstacles}\n"),
    ("requires", "requires:\n  predicate: avoid_sphere\n  warning_r: 0.20\n"
                 "  hard_r: 0.10\n  push_hard: 0.08\n"),
])
def test_missing_required_mapping_key(key, block):
    fm = FRONT_MATTER.replace(block, "")
    assert block not in fm
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert f"{key}: missing required key" in str(exc.value)


@pytest.mark.parametrize("value", ["2", "1.0", "true", '"1"'])
def test_bad_schema_version(tmp_path, value):
    fm = replace("schema_version:", f"schema_version: {value}")
    with pytest.raises(InvalidRuleError, match="schema_version"):
        parse_skill_md(make_text(fm), "rec.md")


@pytest.mark.parametrize("value", ["1.5", "true", '"3"'])
def test_non_integer_revision(tmp_path, value):
    fm = replace("revision:", f"revision: {value}")
    with pytest.raises(InvalidRuleError, match="revision"):
        parse_skill_md(make_text(fm), "rec.md")


@pytest.mark.parametrize("value", ["0", "-2"])
def test_non_positive_revision(tmp_path, value):
    fm = replace("revision:", f"revision: {value}")
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "revision" in str(exc.value) and "positive" in str(exc.value)


@pytest.mark.parametrize("value", ["Bad_Id", "1_leading_digit", "_underscore",
                                   "has-dash", '""', "trailing space"])
def test_id_pattern(tmp_path, value):
    fm = replace("id:", f"id: {value}")
    with pytest.raises(InvalidRuleError, match="id"):
        parse_skill_md(make_text(fm), "rec.md")


@pytest.mark.parametrize("key,value", [
    ("violation_action", "warn"),
    ("unknown_action", "carry_on"),
])
def test_unknown_action_enums(tmp_path, key, value):
    fm = replace(f"{key}:", f"{key}: {value}")
    with pytest.raises(InvalidRuleError, match=key):
        parse_skill_md(make_text(fm), "rec.md")


# ---- roles ---------------------------------------------------------------

def test_roles_not_a_mapping(tmp_path):
    fm = replace("roles:", "roles: [subject]").replace(
        "  subject: {binding: eef}\n", "").replace(
        "  target: {binding: obstacles}\n", "")
    with pytest.raises(InvalidRuleError, match="roles"):
        parse_skill_md(make_text(fm), "rec.md")


def test_missing_subject_role(tmp_path):
    fm = FRONT_MATTER.replace("  subject: {binding: eef}\n", "")
    with pytest.raises(InvalidRuleError, match="roles.subject"):
        parse_skill_md(make_text(fm), "rec.md")


def test_unknown_key_inside_roles(tmp_path):
    fm = FRONT_MATTER.replace(
        "  target: {binding: obstacles}\n",
        "  target: {binding: obstacles}\n  bystander: {binding: obstacles}\n",
    )
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "roles" in str(exc.value) and "bystander" in str(exc.value)


def test_unknown_key_inside_a_role(tmp_path):
    fm = FRONT_MATTER.replace(
        "  subject: {binding: eef}", "  subject: {binding: eef, weight: 2}"
    )
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "roles.subject" in str(exc.value) and "weight" in str(exc.value)


def test_unknown_subject_binding(tmp_path):
    fm = FRONT_MATTER.replace("{binding: eef}", "{binding: elbow}")
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "roles.subject.binding" in str(exc.value)


def test_unknown_target_binding(tmp_path):
    fm = FRONT_MATTER.replace("{binding: obstacles}", "{binding: everything}")
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "roles.target.binding" in str(exc.value)


def test_objects_with_fact_target_binding_is_accepted(tmp_path):
    fm = FRONT_MATTER.replace(
        "{binding: obstacles}", "{binding: 'objects_with_fact:IS_FRAGILE'}"
    )
    record = parse_skill_md(make_text(fm), "rec.md")
    assert record.target_binding == "objects_with_fact:IS_FRAGILE"


def test_objects_with_fact_unknown_predicate(tmp_path):
    fm = FRONT_MATTER.replace(
        "{binding: obstacles}", "{binding: 'objects_with_fact:OVERHEAD_SENSITIVE'}"
    )
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "roles.target.binding" in str(exc.value)
    assert "OVERHEAD_SENSITIVE" in str(exc.value)


def test_objects_with_fact_must_be_upper_case(tmp_path):
    fm = FRONT_MATTER.replace(
        "{binding: obstacles}", "{binding: 'objects_with_fact:is_fragile'}"
    )
    with pytest.raises(InvalidRuleError, match="roles.target.binding"):
        parse_skill_md(make_text(fm), "rec.md")


# ---- requires ------------------------------------------------------------

def test_unknown_effect_predicate(tmp_path):
    fm = FRONT_MATTER.replace("predicate: avoid_sphere", "predicate: avoid_cube")
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "requires.predicate" in str(exc.value)


def test_missing_effect_predicate(tmp_path):
    fm = FRONT_MATTER.replace("  predicate: avoid_sphere\n", "")
    with pytest.raises(InvalidRuleError, match="requires.predicate"):
        parse_skill_md(make_text(fm), "rec.md")


def test_more_than_one_effect(tmp_path):
    fm = FRONT_MATTER.replace(
        "  push_hard: 0.08\n", "  push_hard: 0.08\n  v_max: 0.3\n"
    )
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "requires" in str(exc.value)
    assert "exactly one effect" in str(exc.value)
    assert "v_max" in str(exc.value)


def test_unknown_key_inside_requires(tmp_path):
    fm = FRONT_MATTER.replace(
        "  push_hard: 0.08\n", "  push_hard: 0.08\n  wraning_r: 0.2\n"
    )
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "wraning_r" in str(exc.value)


@pytest.mark.parametrize("param", ["warning_r", "hard_r", "push_hard"])
def test_missing_required_effect_param(tmp_path, param):
    fm = replace(f"  {param}:", None)
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert f"requires.{param}" in str(exc.value)


@pytest.mark.parametrize("line,new,param", [
    ("  warning_r:", "  warning_r: 0.7", "warning_r"),
    ("  warning_r:", "  warning_r: 0.0", "warning_r"),
    ("  hard_r:", "  hard_r: 0.25", "hard_r"),      # > warning_r (0.20)
    ("  hard_r:", "  hard_r: 0.0", "hard_r"),
    ("  push_hard:", "  push_hard: 0.31", "push_hard"),
    ("  push_hard:", "  push_hard: -0.01", "push_hard"),
])
def test_out_of_range_avoid_sphere_param(tmp_path, line, new, param):
    fm = replace(line, new)
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert f"requires.{param}" in str(exc.value) and "outside" in str(exc.value)


def test_out_of_range_cancel_scale_on_target_corridor(tmp_path):
    fm = FRONT_MATTER.replace(
        "  push_hard: 0.08\n",
        "  push_hard: 0.08\n  cancel_scale_on_target_corridor: 1.4\n",
    )
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "cancel_scale_on_target_corridor" in str(exc.value)


def _effect_fm(effect_lines: str, subject: str = "currently_grasped",
               formula: str = "IS_SPILLABLE(SUBJECT)") -> str:
    return (
        "schema_version: 1\n"
        "id: some_rule\n"
        "revision: 1\n"
        "description: d\n"
        "roles:\n"
        f"  subject: {{binding: {subject}}}\n"
        f'applies_when: "{formula}"\n'
        "requires:\n"
        f"{effect_lines}"
        "violation_action: slow\n"
        "unknown_action: ignore\n"
    )


@pytest.mark.parametrize("effect_lines,param", [
    ("  predicate: limit_speed\n  v_max: 1.5\n", "v_max"),
    ("  predicate: limit_speed\n  v_max: 0.0\n", "v_max"),
    ("  predicate: limit_angular_rate\n  omega_max: 5.5\n", "omega_max"),
    ("  predicate: limit_angular_rate\n  omega_max: 0.0\n", "omega_max"),
    ("  predicate: avoid_region_above\n  clearance_m: 0.2\n", "clearance_m"),
    ("  predicate: avoid_region_above\n  clearance_m: -0.01\n", "clearance_m"),
    ("  predicate: avoid_region_above\n  clearance_m: 0.02\n  z_max: 0.5\n", "z_max"),
    ("  predicate: avoid_region_above\n  clearance_m: 0.02\n  z_max: 3.5\n", "z_max"),
    ("  predicate: avoid_region_above\n  clearance_m: 0.02\n  cancel_scale: 0.0\n",
     "cancel_scale"),
    ("  predicate: avoid_region_above\n  clearance_m: 0.02\n  cancel_scale: 1.2\n",
     "cancel_scale"),
])
def test_out_of_range_other_effect_params(effect_lines, param):
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(_effect_fm(effect_lines)), "rec.md")
    assert f"requires.{param}" in str(exc.value) and "outside" in str(exc.value)


@pytest.mark.parametrize("effect_lines,param", [
    ("  predicate: limit_speed\n", "v_max"),
    ("  predicate: limit_angular_rate\n", "omega_max"),
    ("  predicate: avoid_region_above\n", "clearance_m"),
])
def test_missing_required_param_of_other_effects(effect_lines, param):
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(_effect_fm(effect_lines)), "rec.md")
    assert f"requires.{param}" in str(exc.value) and "missing" in str(exc.value)


@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf"])
def test_non_finite_numeric(value):
    lines = f"  predicate: limit_speed\n  v_max: {value}\n"
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(_effect_fm(lines)), "rec.md")
    assert "requires.v_max" in str(exc.value) and "finite" in str(exc.value)


def test_bool_where_a_number_is_required():
    lines = "  predicate: limit_speed\n  v_max: true\n"
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(_effect_fm(lines)), "rec.md")
    assert "requires.v_max" in str(exc.value) and "boolean" in str(exc.value)


def test_string_where_a_number_is_required():
    lines = '  predicate: limit_speed\n  v_max: "0.3"\n'
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(_effect_fm(lines)), "rec.md")
    assert "requires.v_max" in str(exc.value) and "number" in str(exc.value)


def test_bad_shape_enum(tmp_path):
    fm = FRONT_MATTER.replace(
        "  push_hard: 0.08\n", "  push_hard: 0.08\n  shape: superquadric\n"
    )
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "requires.shape" in str(exc.value)


# ---- applies_when --------------------------------------------------------

@pytest.mark.parametrize("formula", [
    "NEAR(SUBJECT, TARGET, 0.20",   # unbalanced: parser exhausts its tokens
    "",                             # empty
    "   ",
])
def test_unparseable_formula(formula):
    fm = replace("applies_when:", f'applies_when: "{formula}"')
    with pytest.raises(InvalidRuleError, match="applies_when"):
        parse_skill_md(make_text(fm), "rec.md")


def test_unknown_predicate_in_formula():
    fm = replace("applies_when:", 'applies_when: "IS_SPICY(SUBJECT) AND NEAR(SUBJECT, TARGET, 0.2)"')
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "applies_when" in str(exc.value) and "IS_SPICY" in str(exc.value)


def test_undeclared_role_token_in_formula():
    fm = replace("applies_when:", 'applies_when: "NEAR(SUBJECT, OTHER, 0.2) AND NEAR(SUBJECT, TARGET, 0.2)"')
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "applies_when" in str(exc.value) and "OTHER" in str(exc.value)


def test_target_referenced_without_a_target_role():
    fm = FRONT_MATTER.replace("  target: {binding: obstacles}\n", "")
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "applies_when" in str(exc.value) and "TARGET" in str(exc.value)


def test_declared_target_role_never_referenced():
    fm = replace("applies_when:", 'applies_when: "NEAR_EDGE()"')
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "roles.target" in str(exc.value)


def test_eef_is_an_allowed_bare_identifier():
    fm = replace("applies_when:", 'applies_when: "NEAR(eef, TARGET, 0.20)"')
    record = parse_skill_md(make_text(fm), "rec.md")
    assert record.applies_when == "NEAR(eef, TARGET, 0.20)"


def test_nested_and_negated_formula_is_walked():
    formula = "NOT (NEAR(SUBJECT, TARGET, 0.2) OR IS_SPICY(TARGET))"
    fm = replace("applies_when:", f'applies_when: "{formula}"')
    with pytest.raises(InvalidRuleError) as exc:
        parse_skill_md(make_text(fm), "rec.md")
    assert "IS_SPICY" in str(exc.value)
