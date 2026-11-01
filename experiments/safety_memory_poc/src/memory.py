"""Persistent safety-fact store.

Stores FACTS about object instances, never rules. The rule set is fixed and
human-written (see rules.py). This separation is the scope boundary of the
proof of life: nothing here induces, generalises, or edits a rule.

Backed by a JSON file so persistence across a full process restart is a real
property that a fixture can test, not an in-memory variable.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Fact:
    """One persisted belief about one object instance."""
    predicate: str          # e.g. "MARKED"
    instance: str           # object instance id, stable across episode resets
    value: bool
    since_episode: str      # episode id in which this was established
    since_t: int            # timestep within that episode
    evidence: str           # contact/event id that produced it
    source: str = "contact_event"

    def key(self) -> str:
        return f"{self.predicate}:{self.instance}"


class SafetyMemory:
    """A durable key-value store of safety facts, keyed by (predicate, instance)."""

    def __init__(self, path: str):
        self.path = path
        self._facts: Dict[str, Fact] = {}
        self._load()

    # ---- durability ----------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as fh:
            blob = json.load(fh)
        if blob.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"memory schema mismatch: file has {blob.get('schema_version')}, "
                f"code expects {SCHEMA_VERSION}")
        self._facts = {k: Fact(**v) for k, v in blob.get("facts", {}).items()}

    def flush(self) -> None:
        """Atomic write, so a killed job cannot leave a half-written store."""
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        blob = {"schema_version": SCHEMA_VERSION,
                "facts": {k: asdict(v) for k, v in self._facts.items()}}
        d = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(blob, fh, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ---- writes ---------------------------------------------------------
    def write(self, fact: Fact) -> None:
        self._facts[fact.key()] = fact
        self.flush()

    def retract(self, predicate: str, instance: str) -> bool:
        """Remove a fact. Used by the wash operator and the deletion fixture."""
        k = f"{predicate}:{instance}"
        existed = k in self._facts
        self._facts.pop(k, None)
        self.flush()
        return existed

    # ---- reads ----------------------------------------------------------
    def get(self, predicate: str, instance: str) -> Optional[Fact]:
        return self._facts.get(f"{predicate}:{instance}")

    def holds(self, predicate: str, instance: str) -> bool:
        f = self.get(predicate, instance)
        return bool(f and f.value)

    def instances_where(self, predicate: str) -> List[str]:
        return sorted(f.instance for f in self._facts.values()
                      if f.predicate == predicate and f.value)

    def all_facts(self) -> List[Fact]:
        return sorted(self._facts.values(), key=lambda f: f.key())

    def __len__(self) -> int:
        return len(self._facts)
