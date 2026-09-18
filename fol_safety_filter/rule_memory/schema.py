"""
FOL Safety Filter — rule-record contract, schema version 1.

A rule record is a ``SKILL.md`` file: one YAML front-matter block (the
executable contract) followed by a Markdown body (rationale only — the body is
never executed and never parsed for semantics).

Validation is strict and all-or-nothing on purpose. The failure mode this
schema refuses is the one `CLAUDE.md` warns about for
`vlm_pipeline/semantic_cbf_filter.py`: a present-but-broken dependency that
silently downgrades the filter instead of raising, and so "changes every
reported number" without anything looking wrong. A malformed record therefore
raises `InvalidRuleError` rather than being skipped, and the loader
(`store.py`) fails the whole snapshot.

One record carries exactly one activation condition and exactly one effect.
That restriction is the decomposition: two effects in one record would recreate
the monolithic obstacle-plus-speed-plus-rotation rule this work exists to take
apart.

`applies_when` is a formula in the **existing** `kb.py` mini-language, parsed by
`kb.parse_formula`. No second formula language is introduced here. `SUBJECT` and
`TARGET` are the only reserved role tokens; `binding.py` substitutes them.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

import yaml

from ..kb import parse_formula
from ..primitives import PREDICATE_REGISTRY

SCHEMA_VERSION = 1
MAX_RECORD_BYTES = 16384

ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

VIOLATION_ACTIONS = ("block", "avoid", "slow", "ask")
UNKNOWN_ACTIONS = ("stop_and_report", "defer_and_refresh", "ignore")

REQUIRED_KEYS = (
    "schema_version",
    "id",
    "revision",
    "description",
    "roles",
    "applies_when",
    "requires",
    "violation_action",
    "unknown_action",
)

SUBJECT_BINDINGS = ("eef", "arm_checkpoints", "currently_grasped")
TARGET_BINDINGS = ("obstacles",)
TARGET_FACT_PREFIX = "objects_with_fact:"

ROLE_TOKENS = ("SUBJECT", "TARGET")
# Bare identifiers a formula argument may name, beyond the two role tokens.
# `eef` is the symbol `primitives.NEAR` and friends special-case to
# `state.ee_pos`, so a record may reference it directly.
LITERAL_SYMBOLS = ("eef",)


class InvalidRuleError(Exception):
    """A rule record (or manifest) violates the version-1 contract."""


# ---------------------------------------------------------------------------
# Effect parameter table
# ---------------------------------------------------------------------------

# Per effect predicate, the numeric parameters in *validation order*. Order is
# load-bearing for `avoid_sphere`: `hard_r`'s upper bound is `warning_r`, so
# `warning_r` must be checked first or a bad `warning_r` surfaces as a
# confusing `hard_r` range failure.
#
# Tuple layout: (name, required, default, lo, lo_inclusive, hi, hi_inclusive).
# A string `hi` names another parameter of the same effect.
_NumSpec = Tuple[str, bool, Optional[float], float, bool, Union[float, str], bool]

EFFECT_NUMERIC_PARAMS: Dict[str, Tuple[_NumSpec, ...]] = {
    "avoid_sphere": (
        ("warning_r", True, None, 0.0, False, 0.60, True),
        ("hard_r", True, None, 0.0, False, "warning_r", True),
        ("push_hard", True, None, 0.0, True, 0.30, True),
        ("cancel_scale_on_target_corridor", False, 1.0, 0.0, False, 1.0, True),
    ),
    "limit_speed": (
        ("v_max", True, None, 0.0, False, 1.0, True),
    ),
    "limit_angular_rate": (
        ("omega_max", True, None, 0.0, False, 5.0, True),
    ),
    "avoid_region_above": (
        ("clearance_m", True, None, 0.0, True, 0.10, True),
        ("z_max", False, 1.40, 0.81, False, 3.0, True),
        ("cancel_scale", False, 1.0, 0.0, False, 1.0, True),
    ),
}

# (name, required, default, allowed values)
_EnumSpec = Tuple[str, bool, Optional[str], Tuple[str, ...]]

EFFECT_ENUM_PARAMS: Dict[str, Tuple[_EnumSpec, ...]] = {
    "avoid_sphere": (
        ("shape", False, "sphere", ("sphere", "ellipsoid_if_available")),
    ),
    "limit_speed": (),
    "limit_angular_rate": (),
    "avoid_region_above": (),
}

EFFECT_PREDICATES = tuple(sorted(EFFECT_NUMERIC_PARAMS))


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EffectSpec:
    """The single effect a record requires, with optional params defaulted in."""

    predicate: str
    params: Mapping[str, Union[float, str]]


@dataclass(frozen=True)
class RuleRecord:
    id: str
    revision: int
    description: str
    roles: Mapping[str, Mapping[str, str]]
    applies_when: str
    requires: EffectSpec
    violation_action: str
    unknown_action: str
    source_path: str
    front_matter_hash: str
    body_hash: str

    @property
    def has_target(self) -> bool:
        return "target" in self.roles

    @property
    def subject_binding(self) -> str:
        return self.roles["subject"]["binding"]

    @property
    def target_binding(self) -> Optional[str]:
        role = self.roles.get("target")
        return role["binding"] if role else None


# ---------------------------------------------------------------------------
# Front-matter splitting
# ---------------------------------------------------------------------------

# `yaml.safe_load` happily resolves anchors and aliases and there is no loader
# flag to forbid them, so they are rejected by a conservative textual pre-scan
# of the raw front matter (never the body) before anything is loaded. A sigil
# only counts at a token start, and `!` only when followed by a non-space, so
# prose like "the ! marker" or "2 * 3" does not trip it.
_ANCHOR_RE = re.compile(r"(?:^|[\s\[\{,])&\S")
_ALIAS_RE = re.compile(r"(?:^|[\s\[\{,])\*\S")
_TAG_RE = re.compile(r"(?:^|[\s\[\{,])!\S")


def split_front_matter(text: str, path: Any) -> Tuple[str, str]:
    """Return ``(raw_front_matter, body)`` as exact substrings of ``text``.

    The `---` delimiter lines are excluded from both. A record must open with a
    delimiter on its first line. A body whose first non-blank line is `---` is
    read as a second front-matter block and rejected; a `---` further down the
    body is an ordinary Markdown horizontal rule and is allowed.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise InvalidRuleError(
            f"{path}: front_matter: missing YAML front matter "
            "(the file must open with a '---' line)"
        )

    close = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close = i
            break
    if close is None:
        raise InvalidRuleError(
            f"{path}: front_matter: unterminated YAML front matter "
            "(no closing '---' line)"
        )

    raw_fm = "".join(lines[1:close])
    body = "".join(lines[close + 1:])

    for line in body.splitlines():
        if not line.strip():
            continue
        if line.strip() == "---":
            raise InvalidRuleError(
                f"{path}: front_matter: more than one front-matter block"
            )
        break

    return raw_fm, body


def _load_front_matter(raw_fm: str, path: Any) -> Dict[str, Any]:
    for label, pattern in (("anchor", _ANCHOR_RE), ("alias", _ALIAS_RE),
                           ("tag", _TAG_RE)):
        if pattern.search(raw_fm):
            raise InvalidRuleError(
                f"{path}: front_matter: YAML {label}s are not permitted"
            )
    try:
        data = yaml.safe_load(raw_fm)
    except yaml.YAMLError as exc:
        raise InvalidRuleError(f"{path}: front_matter: unparseable YAML: {exc}")
    if not isinstance(data, dict):
        raise InvalidRuleError(
            f"{path}: front_matter: expected a mapping, got "
            f"{type(data).__name__}"
        )
    return data


# ---------------------------------------------------------------------------
# Scalar checks
# ---------------------------------------------------------------------------

def _check_str(value: Any, field: str, path: Any) -> str:
    if not isinstance(value, str):
        raise InvalidRuleError(
            f"{path}: {field}: expected a string, got {type(value).__name__}"
        )
    return value


def _check_int(value: Any, field: str, path: Any) -> int:
    # `isinstance(True, int)` is True and `yaml.safe_load("1.0")` gives a
    # float that compares equal to 1, so both are rejected explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidRuleError(
            f"{path}: {field}: expected an integer, got {type(value).__name__}"
        )
    return value


def _check_bool(value: Any, field: str, path: Any) -> bool:
    if not isinstance(value, bool):
        raise InvalidRuleError(
            f"{path}: {field}: expected a boolean, got {type(value).__name__}"
        )
    return value


def _check_number(value: Any, field: str, path: Any) -> float:
    if isinstance(value, bool):
        raise InvalidRuleError(
            f"{path}: {field}: expected a number, got a boolean"
        )
    if not isinstance(value, (int, float)):
        raise InvalidRuleError(
            f"{path}: {field}: expected a number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise InvalidRuleError(f"{path}: {field}: must be finite, got {value!r}")
    return number


def _check_range(number: float, field: str, path: Any,
                 lo: float, lo_inclusive: bool,
                 hi: float, hi_inclusive: bool) -> float:
    low_ok = number >= lo if lo_inclusive else number > lo
    high_ok = number <= hi if hi_inclusive else number < hi
    if not (low_ok and high_ok):
        bounds = f"{'[' if lo_inclusive else '('}{lo}, {hi}{']' if hi_inclusive else ')'}"
        raise InvalidRuleError(
            f"{path}: requires.{field}: {number!r} is outside {bounds}"
        )
    return number


def _check_no_extra_keys(data: Mapping[str, Any], allowed: Tuple[str, ...],
                         field: str, path: Any) -> None:
    extra = sorted(k for k in data if k not in allowed)
    if extra:
        raise InvalidRuleError(
            f"{path}: {field}: unknown key(s) {extra}; permitted: "
            f"{sorted(allowed)}"
        )


# ---------------------------------------------------------------------------
# Section validators
# ---------------------------------------------------------------------------

def _validate_roles(raw: Any, path: Any) -> Dict[str, Dict[str, str]]:
    if not isinstance(raw, dict):
        raise InvalidRuleError(
            f"{path}: roles: expected a mapping, got {type(raw).__name__}"
        )
    _check_no_extra_keys(raw, ("subject", "target"), "roles", path)
    if "subject" not in raw:
        raise InvalidRuleError(f"{path}: roles.subject: missing required role")

    roles: Dict[str, Dict[str, str]] = {}
    for role in ("subject", "target"):
        if role not in raw:
            continue
        spec = raw[role]
        if not isinstance(spec, dict):
            raise InvalidRuleError(
                f"{path}: roles.{role}: expected a mapping, got "
                f"{type(spec).__name__}"
            )
        _check_no_extra_keys(spec, ("binding",), f"roles.{role}", path)
        if "binding" not in spec:
            raise InvalidRuleError(f"{path}: roles.{role}.binding: missing")
        binding = _check_str(spec["binding"], f"roles.{role}.binding", path)
        _validate_binding(role, binding, path)
        roles[role] = {"binding": binding}
    return roles


def _validate_binding(role: str, binding: str, path: Any) -> None:
    field = f"roles.{role}.binding"
    if role == "subject":
        if binding not in SUBJECT_BINDINGS:
            raise InvalidRuleError(
                f"{path}: {field}: unknown binding {binding!r}; permitted: "
                f"{list(SUBJECT_BINDINGS)}"
            )
        return

    if binding in TARGET_BINDINGS:
        return
    if binding.startswith(TARGET_FACT_PREFIX):
        fact = binding[len(TARGET_FACT_PREFIX):]
        if not fact or fact != fact.upper():
            raise InvalidRuleError(
                f"{path}: {field}: {binding!r} must name an upper-case "
                "predicate after the prefix"
            )
        if fact not in PREDICATE_REGISTRY:
            raise InvalidRuleError(
                f"{path}: {field}: unknown predicate {fact!r} in {binding!r}"
            )
        return
    raise InvalidRuleError(
        f"{path}: {field}: unknown binding {binding!r}; permitted: "
        f"{list(TARGET_BINDINGS)} or '{TARGET_FACT_PREFIX}<FACT>'"
    )


def _validate_requires(raw: Any, path: Any) -> EffectSpec:
    if not isinstance(raw, dict):
        raise InvalidRuleError(
            f"{path}: requires: expected a mapping, got {type(raw).__name__}"
        )
    if "predicate" not in raw:
        raise InvalidRuleError(f"{path}: requires.predicate: missing")
    predicate = _check_str(raw["predicate"], "requires.predicate", path)
    if predicate not in EFFECT_NUMERIC_PARAMS:
        raise InvalidRuleError(
            f"{path}: requires.predicate: unknown effect {predicate!r}; "
            f"permitted: {list(EFFECT_PREDICATES)}"
        )

    numeric = EFFECT_NUMERIC_PARAMS[predicate]
    enums = EFFECT_ENUM_PARAMS[predicate]
    allowed = ("predicate",) + tuple(s[0] for s in numeric) + tuple(s[0] for s in enums)
    extra = sorted(k for k in raw if k not in allowed)
    if extra:
        # An extra key here is either a typo or a second effect's parameter.
        # Both mean the record no longer carries exactly one effect.
        raise InvalidRuleError(
            f"{path}: requires: unknown key(s) {extra} for effect "
            f"{predicate!r} — a record carries exactly one effect; permitted: "
            f"{sorted(allowed)}"
        )

    params: Dict[str, Union[float, str]] = {}
    for name, required, default, lo, lo_inc, hi, hi_inc in numeric:
        if name not in raw:
            if required:
                raise InvalidRuleError(
                    f"{path}: requires.{name}: missing required parameter of "
                    f"effect {predicate!r}"
                )
            params[name] = float(default)
            continue
        number = _check_number(raw[name], f"requires.{name}", path)
        upper = params[hi] if isinstance(hi, str) else hi
        params[name] = _check_range(number, name, path, lo, lo_inc,
                                    float(upper), hi_inc)

    for name, required, default, choices in enums:
        if name not in raw:
            if required:
                raise InvalidRuleError(
                    f"{path}: requires.{name}: missing required parameter of "
                    f"effect {predicate!r}"
                )
            params[name] = default
            continue
        value = _check_str(raw[name], f"requires.{name}", path)
        if value not in choices:
            raise InvalidRuleError(
                f"{path}: requires.{name}: {value!r} is not one of "
                f"{list(choices)}"
            )
        params[name] = value

    # A read-only view, so a frozen `RuleRecord` is immutable transitively and
    # a caller cannot retune a record's geometry in place mid-episode.
    return EffectSpec(predicate=predicate, params=MappingProxyType(dict(params)))


def _iter_atoms(node: Any) -> Iterator[Any]:
    """Yield every atom of a `kb.parse_formula` AST.

    Walked by shape rather than by `isinstance` against `kb._Atom` / `_Not` /
    `_And` / `_Or` so this module does not couple to `kb`'s private class names.
    """
    if hasattr(node, "name") and hasattr(node, "args"):
        yield node
        return
    child = getattr(node, "child", None)
    if child is not None:
        yield from _iter_atoms(child)
        return
    for sub in getattr(node, "children", ()):
        yield from _iter_atoms(sub)


def _validate_formula(formula: str, roles: Mapping[str, Any], path: Any) -> None:
    if not formula.strip():
        raise InvalidRuleError(f"{path}: applies_when: formula is empty")
    try:
        parsed = parse_formula(formula)
    except Exception as exc:
        # The parser raises ValueError on an unexpected token and IndexError on
        # premature token exhaustion, so neither type can be relied on.
        raise InvalidRuleError(
            f"{path}: applies_when: unparseable formula {formula!r}: {exc}"
        )

    used_target = False
    for atom in _iter_atoms(parsed):
        if atom.name.upper() not in PREDICATE_REGISTRY:
            raise InvalidRuleError(
                f"{path}: applies_when: unknown predicate {atom.name!r}"
            )
        for arg in atom.args:
            if not isinstance(arg, str):
                continue
            if arg == "TARGET":
                used_target = True
                continue
            if arg in ROLE_TOKENS or arg in LITERAL_SYMBOLS:
                continue
            raise InvalidRuleError(
                f"{path}: applies_when: undeclared role token {arg!r}; "
                f"permitted: {list(ROLE_TOKENS) + list(LITERAL_SYMBOLS)}"
            )

    if used_target and "target" not in roles:
        raise InvalidRuleError(
            f"{path}: applies_when: references TARGET but roles.target is not "
            "declared"
        )
    if "target" in roles and not used_target:
        raise InvalidRuleError(
            f"{path}: roles.target: declared but TARGET is never referenced in "
            "applies_when"
        )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def parse_skill_md(text: str, path: Any) -> RuleRecord:
    """Parse and fully validate one ``SKILL.md``; raise `InvalidRuleError`."""
    if len(text.encode("utf-8")) > MAX_RECORD_BYTES:
        raise InvalidRuleError(
            f"{path}: file: {len(text.encode('utf-8'))} bytes exceeds the "
            f"{MAX_RECORD_BYTES}-byte limit"
        )

    raw_fm, body = split_front_matter(text, path)
    data = _load_front_matter(raw_fm, path)

    _check_no_extra_keys(data, REQUIRED_KEYS, "front_matter", path)
    for key in REQUIRED_KEYS:
        if key not in data:
            raise InvalidRuleError(f"{path}: {key}: missing required key")

    schema_version = _check_int(data["schema_version"], "schema_version", path)
    if schema_version != SCHEMA_VERSION:
        raise InvalidRuleError(
            f"{path}: schema_version: expected {SCHEMA_VERSION}, got "
            f"{schema_version}"
        )

    rule_id = _check_str(data["id"], "id", path)
    if not ID_PATTERN.match(rule_id):
        raise InvalidRuleError(
            f"{path}: id: {rule_id!r} does not match {ID_PATTERN.pattern}"
        )

    revision = _check_int(data["revision"], "revision", path)
    if revision <= 0:
        raise InvalidRuleError(f"{path}: revision: must be positive, got {revision}")

    description = _check_str(data["description"], "description", path)
    roles = _validate_roles(data["roles"], path)
    applies_when = _check_str(data["applies_when"], "applies_when", path)
    requires = _validate_requires(data["requires"], path)

    violation_action = _check_str(data["violation_action"], "violation_action", path)
    if violation_action not in VIOLATION_ACTIONS:
        raise InvalidRuleError(
            f"{path}: violation_action: {violation_action!r} is not one of "
            f"{list(VIOLATION_ACTIONS)}"
        )
    unknown_action = _check_str(data["unknown_action"], "unknown_action", path)
    if unknown_action not in UNKNOWN_ACTIONS:
        raise InvalidRuleError(
            f"{path}: unknown_action: {unknown_action!r} is not one of "
            f"{list(UNKNOWN_ACTIONS)}"
        )

    _validate_formula(applies_when, roles, path)

    return RuleRecord(
        id=rule_id,
        revision=revision,
        description=description,
        roles=roles,
        applies_when=applies_when,
        requires=requires,
        violation_action=violation_action,
        unknown_action=unknown_action,
        source_path=str(path),
        front_matter_hash=hashlib.sha256(raw_fm.encode("utf-8")).hexdigest(),
        body_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )


def load_record(path: Union[str, Path]) -> RuleRecord:
    """Read one ``SKILL.md`` from disk and parse it."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise InvalidRuleError(f"{p}: file: cannot be read: {exc}")
    if len(raw) > MAX_RECORD_BYTES:
        raise InvalidRuleError(
            f"{p}: file: {len(raw)} bytes exceeds the {MAX_RECORD_BYTES}-byte limit"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidRuleError(f"{p}: file: not valid UTF-8: {exc}")
    return parse_skill_md(text, p)


def validate(record: RuleRecord) -> RuleRecord:
    """Re-run the semantic checks on an already-built `RuleRecord`.

    `parse_skill_md` validates as it builds, so this is for records that arrive
    by some other route (a round-trip, a test fixture) and for asserting that a
    record still satisfies the contract against the current
    `PREDICATE_REGISTRY`.
    """
    path = record.source_path or "<record>"
    if record.requires.predicate not in EFFECT_NUMERIC_PARAMS:
        raise InvalidRuleError(
            f"{path}: requires.predicate: unknown effect "
            f"{record.requires.predicate!r}"
        )
    _validate_roles({k: dict(v) for k, v in record.roles.items()}, path)
    _validate_requires(
        {"predicate": record.requires.predicate, **dict(record.requires.params)},
        path,
    )
    _validate_formula(record.applies_when, record.roles, path)
    return record
