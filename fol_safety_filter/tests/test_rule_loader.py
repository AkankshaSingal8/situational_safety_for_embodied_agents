"""Rule-memory loader: manifest strictness, snapshot selection, and the
reload-identity property `compiled_digest()` exists to make testable."""

from __future__ import annotations

import pytest

from fol_safety_filter.rule_memory import InvalidRuleError, load_library

SPHERE_RECORD = """---
schema_version: 1
id: {rid}
revision: {revision}
description: {rid} keeps the subject clear of each obstacle.
roles:
  subject: {{binding: eef}}
  target: {{binding: obstacles}}
applies_when: "NEAR(SUBJECT, TARGET, 0.20)"
requires:
  predicate: avoid_sphere
  warning_r: 0.20
  hard_r: 0.10
  push_hard: 0.08
violation_action: avoid
unknown_action: stop_and_report
---

Geometric proximity only.
"""


def write_record(root, rid, revision=1, dirname=None, text=None):
    directory = root / (dirname or rid)
    directory.mkdir(parents=True, exist_ok=True)
    body = text if text is not None else SPHERE_RECORD.format(rid=rid, revision=revision)
    (directory / "SKILL.md").write_text(body, encoding="utf-8")
    return directory


def write_manifest(root, entries, schema_version=1):
    lines = [f"schema_version: {schema_version}", "rules:"]
    for rid, default_loaded in entries:
        lines.append(f"  - id: {rid}")
        lines.append(f"    default_loaded: {str(default_loaded).lower()}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "MANIFEST.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root / "MANIFEST.yaml"


def build_library(root, entries, revisions=None):
    revisions = revisions or {}
    for rid, _ in entries:
        write_record(root, rid, revision=revisions.get(rid, 1))
    write_manifest(root, entries)
    return root


# ---- selection and ordering ---------------------------------------------

def test_records_load_in_sorted_id_order(tmp_path):
    # Manifest order is deliberately not sorted.
    root = build_library(tmp_path / "mem", [("zebra_rule", True),
                                            ("alpha_rule", True),
                                            ("middle_rule", True)])
    lib = load_library(root)
    assert [r.id for r in lib.records] == ["alpha_rule", "middle_rule", "zebra_rule"]
    assert [r.id for r in lib.all_records] == ["alpha_rule", "middle_rule", "zebra_rule"]
    assert lib.manifest_path == str(root / "MANIFEST.yaml")


def test_default_loaded_selects_the_enabled_subset(tmp_path):
    root = build_library(tmp_path / "mem", [("a_rule", True),
                                            ("b_rule", False),
                                            ("c_rule", True)])
    lib = load_library(root)
    assert lib.enabled_ids() == ("a_rule", "c_rule")
    # Opt-in records are still loaded and validated, just not enabled.
    assert [r.id for r in lib.all_records] == ["a_rule", "b_rule", "c_rule"]


def test_enabled_ids_overrides_the_manifest(tmp_path):
    root = build_library(tmp_path / "mem", [("a_rule", True), ("b_rule", False)])
    lib = load_library(root, enabled_ids=["b_rule"])
    assert lib.enabled_ids() == ("b_rule",)
    assert len(lib.all_records) == 2


def test_enabled_ids_may_be_empty(tmp_path):
    root = build_library(tmp_path / "mem", [("a_rule", True)])
    lib = load_library(root, enabled_ids=[])
    assert lib.records == ()
    assert len(lib.all_records) == 1


def test_enabled_ids_must_be_a_manifest_subset(tmp_path):
    root = build_library(tmp_path / "mem", [("a_rule", True)])
    with pytest.raises(InvalidRuleError) as exc:
        load_library(root, enabled_ids=["a_rule", "ghost_rule"])
    assert "enabled_ids" in str(exc.value) and "ghost_rule" in str(exc.value)


def test_record_by_id_sees_only_enabled_records(tmp_path):
    root = build_library(tmp_path / "mem", [("a_rule", True), ("b_rule", False)])
    lib = load_library(root)
    assert lib.record_by_id("a_rule").id == "a_rule"
    with pytest.raises(KeyError):
        lib.record_by_id("b_rule")


def test_library_is_frozen(tmp_path):
    root = build_library(tmp_path / "mem", [("a_rule", True)])
    lib = load_library(root)
    with pytest.raises(Exception):
        lib.records = ()


# ---- all-or-nothing ------------------------------------------------------

def test_one_invalid_record_fails_the_whole_snapshot(tmp_path):
    root = tmp_path / "mem"
    write_record(root, "good_rule")
    write_record(root, "bad_rule",
                 text=SPHERE_RECORD.format(rid="bad_rule", revision=1).replace(
                     "hard_r: 0.10", "hard_r: 0.90"))
    write_manifest(root, [("good_rule", True), ("bad_rule", True)])
    with pytest.raises(InvalidRuleError) as exc:
        load_library(root)
    assert "hard_r" in str(exc.value)


def test_an_invalid_opt_in_record_also_fails_the_snapshot(tmp_path):
    root = tmp_path / "mem"
    write_record(root, "good_rule")
    write_record(root, "bad_rule",
                 text=SPHERE_RECORD.format(rid="bad_rule", revision=1).replace(
                     "predicate: avoid_sphere", "predicate: avoid_cube"))
    write_manifest(root, [("good_rule", True), ("bad_rule", False)])
    with pytest.raises(InvalidRuleError, match="requires.predicate"):
        load_library(root)


def test_manifested_id_without_a_record_file(tmp_path):
    root = tmp_path / "mem"
    write_record(root, "present_rule")
    write_manifest(root, [("present_rule", True), ("absent_rule", True)])
    with pytest.raises(InvalidRuleError) as exc:
        load_library(root)
    assert "absent_rule" in str(exc.value)


def test_record_dir_absent_from_the_manifest(tmp_path):
    root = tmp_path / "mem"
    write_record(root, "listed_rule")
    write_record(root, "stowaway_rule")
    write_manifest(root, [("listed_rule", True)])
    with pytest.raises(InvalidRuleError) as exc:
        load_library(root)
    assert "stowaway_rule" in str(exc.value)
    assert "absent from the manifest" in str(exc.value)


def test_non_record_directories_are_ignored(tmp_path):
    root = build_library(tmp_path / "mem", [("a_rule", True)])
    (root / "fixtures").mkdir()
    (root / "fixtures" / "scene_facts.yaml").write_text("scenes: []\n", encoding="utf-8")
    lib = load_library(root)
    assert lib.enabled_ids() == ("a_rule",)


def test_id_dirname_mismatch(tmp_path):
    root = tmp_path / "mem"
    write_record(root, "declared_rule", dirname="manifest_rule")
    write_manifest(root, [("manifest_rule", True)])
    with pytest.raises(InvalidRuleError) as exc:
        load_library(root)
    assert "declared_rule" in str(exc.value) and "manifest_rule" in str(exc.value)


def test_missing_manifest(tmp_path):
    root = tmp_path / "mem"
    write_record(root, "a_rule")
    with pytest.raises(InvalidRuleError, match="manifest"):
        load_library(root)


def test_manifest_schema_version(tmp_path):
    root = build_library(tmp_path / "mem", [("a_rule", True)])
    write_manifest(root, [("a_rule", True)], schema_version=2)
    with pytest.raises(InvalidRuleError, match="schema_version"):
        load_library(root)


def test_manifest_unknown_top_level_key(tmp_path):
    root = build_library(tmp_path / "mem", [("a_rule", True)])
    (root / "MANIFEST.yaml").write_text(
        "schema_version: 1\nowner: me\nrules:\n  - id: a_rule\n"
        "    default_loaded: true\n", encoding="utf-8")
    with pytest.raises(InvalidRuleError) as exc:
        load_library(root)
    assert "owner" in str(exc.value)


def test_manifest_unknown_entry_key(tmp_path):
    root = build_library(tmp_path / "mem", [("a_rule", True)])
    (root / "MANIFEST.yaml").write_text(
        "schema_version: 1\nrules:\n  - id: a_rule\n    default_loaded: true\n"
        "    priority: 3\n", encoding="utf-8")
    with pytest.raises(InvalidRuleError) as exc:
        load_library(root)
    assert "priority" in str(exc.value)


def test_manifest_default_loaded_must_be_a_bool(tmp_path):
    root = build_library(tmp_path / "mem", [("a_rule", True)])
    (root / "MANIFEST.yaml").write_text(
        "schema_version: 1\nrules:\n  - id: a_rule\n    default_loaded: yes_please\n",
        encoding="utf-8")
    with pytest.raises(InvalidRuleError, match="default_loaded"):
        load_library(root)


def test_manifest_duplicate_id(tmp_path):
    root = build_library(tmp_path / "mem", [("a_rule", True)])
    (root / "MANIFEST.yaml").write_text(
        "schema_version: 1\nrules:\n  - id: a_rule\n    default_loaded: true\n"
        "  - id: a_rule\n    default_loaded: false\n", encoding="utf-8")
    with pytest.raises(InvalidRuleError) as exc:
        load_library(root)
    assert "duplicate id" in str(exc.value)


# ---- compiled digest -----------------------------------------------------

def test_compiled_digest_is_stable_across_reloads(tmp_path):
    root = build_library(tmp_path / "mem", [("a_rule", True), ("b_rule", True)])
    first = load_library(root)
    second = load_library(root)
    assert first.compiled_digest() == second.compiled_digest()


def test_compiled_digest_ignores_the_absolute_path(tmp_path):
    entries = [("a_rule", True), ("b_rule", True)]
    here = build_library(tmp_path / "here", entries)
    there = build_library(tmp_path / "elsewhere" / "nested", entries)
    assert load_library(here).compiled_digest() == load_library(there).compiled_digest()
    assert load_library(here).manifest_path != load_library(there).manifest_path


def test_compiled_digest_changes_with_a_revision(tmp_path):
    entries = [("a_rule", True), ("b_rule", True)]
    before = build_library(tmp_path / "before", entries)
    after = build_library(tmp_path / "after", entries, revisions={"b_rule": 2})
    assert load_library(before).compiled_digest() != load_library(after).compiled_digest()


def test_compiled_digest_covers_only_enabled_records(tmp_path):
    root = build_library(tmp_path / "mem", [("a_rule", True), ("b_rule", False)])
    lib = load_library(root)
    only_a = load_library(root, enabled_ids=["a_rule"])
    both = load_library(root, enabled_ids=["a_rule", "b_rule"])
    assert lib.compiled_digest() == only_a.compiled_digest()
    assert lib.compiled_digest() != both.compiled_digest()


def test_a_mid_episode_edit_does_not_change_the_snapshot(tmp_path):
    root = build_library(tmp_path / "mem", [("a_rule", True)])
    lib = load_library(root)
    digest = lib.compiled_digest()
    revision = lib.record_by_id("a_rule").revision

    write_record(root, "a_rule", revision=7)
    assert lib.compiled_digest() == digest
    assert lib.record_by_id("a_rule").revision == revision
    # The edit lands at the next load, i.e. the next episode boundary.
    assert load_library(root).compiled_digest() != digest
