"""
FOL Safety Filter — on-disk rule-memory loader.

`load_library` reads a `MANIFEST.yaml`, loads and validates **every** record it
names, and only then decides which of them are enabled. Validating all of them
regardless of `default_loaded` is deliberate: an opt-in record that is broken
must be caught at load time, not the first time somebody enables it.

Loading is a snapshot. A `RuleLibrary` is frozen and exposes no mutating API, so
editing a `SKILL.md` mid-episode cannot change the rules already in play — the
edit takes effect at the next `load_library` call, i.e. the next episode
boundary. That is by construction, not by a watchdog.

Manifest completeness is checked in both directions: a manifested id whose
`<id>/SKILL.md` is missing is invalid, and a subdirectory that *contains* a
`SKILL.md` but is absent from the manifest is invalid. Only directories holding
a literal `SKILL.md` count as record directories, so sibling data directories
(e.g. the `fixtures/` tree the design puts alongside the records) are ignored
rather than rejected. The consequence: a record directory whose file is
misspelled (`skill.md`, `SKILL.MD`) is silently ignored instead of flagged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import yaml

from .schema import (
    SCHEMA_VERSION,
    InvalidRuleError,
    RuleRecord,
    _check_bool,
    _check_int,
    _check_no_extra_keys,
    _check_str,
    load_record,
)

MANIFEST_NAME = "MANIFEST.yaml"
RECORD_NAME = "SKILL.md"


@dataclass(frozen=True)
class RuleLibrary:
    """An immutable snapshot of a rule memory."""

    records: Tuple[RuleRecord, ...]      # enabled, sorted by id
    all_records: Tuple[RuleRecord, ...]  # every manifested record, sorted by id
    manifest_path: str

    def record_by_id(self, rule_id: str) -> RuleRecord:
        """Return the *enabled* record with this id, or raise `KeyError`.

        Disabled-but-manifested records are reachable through `all_records`
        only; looking one up here is treated as a caller error rather than
        quietly handing back a rule that is not in the snapshot.
        """
        for record in self.records:
            if record.id == rule_id:
                return record
        raise KeyError(rule_id)

    def enabled_ids(self) -> Tuple[str, ...]:
        return tuple(r.id for r in self.records)

    def compiled_digest(self) -> str:
        """A stable digest of the enabled snapshot.

        Covers each enabled record's `(id, revision, front_matter_hash)` in
        sorted id order, and nothing else — no absolute path, no mapping
        iteration order, no filesystem timestamp — so two independent loads of
        the same bytes from different locations agree, while any edit to a
        record's contract changes it.
        """
        payload = "\n".join(
            f"{r.id}\t{r.revision}\t{r.front_matter_hash}" for r in self.records
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _read_manifest(manifest_path: Path) -> List[Tuple[str, bool]]:
    if not manifest_path.is_file():
        raise InvalidRuleError(f"{manifest_path}: manifest: file not found")
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise InvalidRuleError(f"{manifest_path}: manifest: unreadable: {exc}")

    if not isinstance(data, dict):
        raise InvalidRuleError(
            f"{manifest_path}: manifest: expected a mapping, got "
            f"{type(data).__name__}"
        )
    _check_no_extra_keys(data, ("schema_version", "rules"), "manifest", manifest_path)
    for key in ("schema_version", "rules"):
        if key not in data:
            raise InvalidRuleError(f"{manifest_path}: {key}: missing required key")

    version = _check_int(data["schema_version"], "schema_version", manifest_path)
    if version != SCHEMA_VERSION:
        raise InvalidRuleError(
            f"{manifest_path}: schema_version: expected {SCHEMA_VERSION}, got "
            f"{version}"
        )

    raw_rules = data["rules"]
    if not isinstance(raw_rules, list):
        raise InvalidRuleError(
            f"{manifest_path}: rules: expected a list, got "
            f"{type(raw_rules).__name__}"
        )

    entries: List[Tuple[str, bool]] = []
    seen: Dict[str, int] = {}
    for i, entry in enumerate(raw_rules):
        field = f"rules[{i}]"
        if not isinstance(entry, dict):
            raise InvalidRuleError(
                f"{manifest_path}: {field}: expected a mapping, got "
                f"{type(entry).__name__}"
            )
        _check_no_extra_keys(entry, ("id", "default_loaded"), field, manifest_path)
        for key in ("id", "default_loaded"):
            if key not in entry:
                raise InvalidRuleError(
                    f"{manifest_path}: {field}.{key}: missing required key"
                )
        rule_id = _check_str(entry["id"], f"{field}.id", manifest_path)
        default_loaded = _check_bool(
            entry["default_loaded"], f"{field}.default_loaded", manifest_path
        )
        if rule_id in seen:
            raise InvalidRuleError(
                f"{manifest_path}: {field}.id: duplicate id {rule_id!r} "
                f"(already at rules[{seen[rule_id]}])"
            )
        seen[rule_id] = i
        entries.append((rule_id, default_loaded))

    return entries


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_library(root, enabled_ids: Optional[Iterable[str]] = None) -> RuleLibrary:
    """Load and validate a rule memory rooted at ``root``.

    `enabled_ids=None` enables the records the manifest marks
    `default_loaded: true`. An explicit `enabled_ids` overrides the manifest
    entirely and must be a subset of the manifested ids.

    Any single invalid manifest entry or record raises `InvalidRuleError` and no
    library is returned — a broken memory never degrades to a partial load.
    """
    root_path = Path(root)
    manifest_path = root_path / MANIFEST_NAME
    entries = _read_manifest(manifest_path)
    manifested = {rule_id for rule_id, _ in entries}

    on_disk = {
        child.name
        for child in sorted(root_path.iterdir())
        if child.is_dir() and (child / RECORD_NAME).is_file()
    }
    unlisted = sorted(on_disk - manifested)
    if unlisted:
        raise InvalidRuleError(
            f"{manifest_path}: rules: record director{'y' if len(unlisted) == 1 else 'ies'} "
            f"{unlisted} present on disk but absent from the manifest"
        )

    records: Dict[str, RuleRecord] = {}
    for rule_id, _ in entries:
        record_path = root_path / rule_id / RECORD_NAME
        if not record_path.is_file():
            raise InvalidRuleError(
                f"{record_path}: rules: manifested id {rule_id!r} has no "
                f"{rule_id}/{RECORD_NAME}"
            )
        record = load_record(record_path)
        if record.id != rule_id:
            raise InvalidRuleError(
                f"{record_path}: id: front-matter id {record.id!r} disagrees "
                f"with directory name {rule_id!r}"
            )
        records[rule_id] = record

    if enabled_ids is None:
        selected = {rule_id for rule_id, default_loaded in entries if default_loaded}
    else:
        selected = set(enabled_ids)
        unknown = sorted(selected - manifested)
        if unknown:
            raise InvalidRuleError(
                f"{manifest_path}: enabled_ids: {unknown} not present in the "
                "manifest"
            )

    all_sorted = tuple(records[rid] for rid in sorted(records))
    enabled_sorted = tuple(r for r in all_sorted if r.id in selected)
    return RuleLibrary(
        records=enabled_sorted,
        all_records=all_sorted,
        manifest_path=str(manifest_path),
    )


def install_rules(kb, scene, root=None, enabled_ids: Optional[Iterable[str]] = None):
    """Load the memory and add its lifted rules to ``kb``. Returns the rules.

    Used by the parity harness and the unit tests. The filter has its own
    loader (`_load_memory_rules`) because it additionally resolves facts from
    the episode's `ObjectState`s and logs each record's revision and hash;
    both go through `load_library` and `resolve`, so neither can end up with a
    different notion of which records are in force.

    Callers that build a filter without ``__init__`` need this because the
    geometry is no longer hardcoded: a knowledge base with no records produces
    no corrections at all. That is the decomposition working, but it makes the
    setup step mandatory rather than optional.

    `root` defaults to the package's own `safety_memory/`.
    """
    from .binding import resolve

    if root is None:
        root = Path(__file__).resolve().parent.parent / "safety_memory"
    library = load_library(root, enabled_ids=enabled_ids)
    rules = []
    for record in library.records:
        for rule in resolve(record, scene):
            kb.add_rule(rule)
            rules.append(rule)
    return rules
