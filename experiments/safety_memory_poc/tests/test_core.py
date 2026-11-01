import os, sys, json, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.ledger import (HazardLedger, ContactEvent, ROLE_TOOL, ROLE_MARKED_SOURCE,
                        ROLE_CLEAN_SOURCE, ROLE_RTE, ROLE_WASH)
from src.memory import SafetyMemory, Fact
from src.rules import ground, KeepAway
from src.cells import belief_source, ARM_MEMORYLESS, ARM_MEMORY, ARM_ORACLE, CELLS

ROLES = {"eef": ROLE_TOOL, "raw": ROLE_MARKED_SOURCE, "fake_raw": ROLE_CLEAN_SOURCE,
         "plate": ROLE_RTE, "sink": ROLE_WASH}


def L():
    return HazardLedger(roles=dict(ROLES))


# ---- ledger: the ground-truth transitions -------------------------------
def test_touching_marked_source_marks_the_tool():
    l = L()
    assert not l.is_marked("eef")
    assert l.step(ContactEvent(1, "eef", "raw")) == "mark"
    assert l.is_marked("eef")


def test_lookalike_contact_is_inert():
    l = L()
    assert l.step(ContactEvent(1, "eef", "fake_raw")) == "clean_contact"
    assert not l.is_marked("eef")


def test_marked_tool_touching_rte_is_a_violation():
    l = L()
    l.step(ContactEvent(1, "eef", "raw"))
    assert l.step(ContactEvent(5, "eef", "plate")) == "violation"
    assert len(l.violations) == 1


def test_unmarked_tool_touching_rte_is_safe():
    l = L()
    assert l.step(ContactEvent(5, "eef", "plate")) == "safe_rte_contact"
    assert l.violations == []


def test_wash_clears_the_mark_and_rte_becomes_safe():
    l = L()
    l.step(ContactEvent(1, "eef", "raw"))
    assert l.step(ContactEvent(2, "eef", "sink")) == "wash"
    assert not l.is_marked("eef")
    assert l.step(ContactEvent(5, "eef", "plate")) == "safe_rte_contact"
    assert l.violations == []


def test_mark_survives_episode_reset_but_violations_do_not():
    """The cross-episode commitment: pose reset must not clean the tool."""
    l = L()
    l.step(ContactEvent(1, "eef", "raw"))
    l.step(ContactEvent(2, "eef", "plate"))
    assert len(l.violations) == 1
    l.reset_episode()
    assert l.violations == []
    assert l.is_marked("eef") is True


def test_order_matters_wash_then_mark_leaves_marked():
    l = L()
    l.step(ContactEvent(1, "eef", "sink"))
    l.step(ContactEvent(2, "eef", "raw"))
    assert l.is_marked("eef")


def test_unrelated_contact_returns_none():
    l = L()
    assert l.step(ContactEvent(1, "raw", "plate")) is None


# ---- memory: durability --------------------------------------------------
def test_memory_roundtrip_and_restart_persistence(tmp_path):
    p = str(tmp_path / "m.json")
    m = SafetyMemory(p)
    m.write(Fact("MARKED", "eef", True, "E0", 3, "contact_7"))
    assert m.holds("MARKED", "eef")
    # A genuinely separate process must see it.
    code = (f"import sys; sys.path.insert(0,{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r});"
            f"from src.memory import SafetyMemory;"
            f"print(SafetyMemory({p!r}).holds('MARKED','eef'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "True", out.stderr


def test_memory_deletion_removes_the_belief(tmp_path):
    p = str(tmp_path / "m.json")
    m = SafetyMemory(p)
    m.write(Fact("MARKED", "eef", True, "E0", 3, "c1"))
    assert m.retract("MARKED", "eef") is True
    assert not m.holds("MARKED", "eef")
    assert SafetyMemory(p).holds("MARKED", "eef") is False


def test_memory_rejects_schema_mismatch(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"schema_version": 999, "facts": {}}))
    with pytest.raises(ValueError):
        SafetyMemory(str(p))


def test_absent_fact_is_false_not_error(tmp_path):
    m = SafetyMemory(str(tmp_path / "m.json"))
    assert m.holds("MARKED", "never_seen") is False


# ---- rule grounding ------------------------------------------------------
def test_rule_grounds_only_when_antecedent_believed():
    assert ground(ROLES, lambda i: False, 0.1, "test") == []
    out = ground(ROLES, lambda i: i == "eef", 0.1, "test")
    assert out == [KeepAway("eef", "plate", 0.1, "test")]


def test_rule_text_is_arm_independent():
    """Same rule, different belief source -> the ONLY difference between arms."""
    marked = ground(ROLES, lambda i: True, 0.1, "a")
    unmarked = ground(ROLES, lambda i: False, 0.1, "b")
    assert len(marked) == 1 and unmarked == []


# ---- arms: the interaction the experiment measures -----------------------
@pytest.mark.parametrize("observable", [True, False])
def test_oracle_always_sees_the_truth(observable, tmp_path):
    l = L(); l.step(ContactEvent(1, "eef", "raw"))
    m = SafetyMemory(str(tmp_path / "m.json"))
    assert belief_source(ARM_ORACLE, l, m, observable)("eef") is True


def test_memoryless_is_blind_when_hidden_and_sighted_when_observable(tmp_path):
    l = L(); l.step(ContactEvent(1, "eef", "raw"))
    m = SafetyMemory(str(tmp_path / "m.json"))
    assert belief_source(ARM_MEMORYLESS, l, m, False)("eef") is False
    assert belief_source(ARM_MEMORYLESS, l, m, True)("eef") is True


def test_memory_recovers_the_hidden_fact(tmp_path):
    l = L(); l.step(ContactEvent(1, "eef", "raw"))
    m = SafetyMemory(str(tmp_path / "m.json"))
    m.write(Fact("MARKED", "eef", True, "E0", 1, "c1"))
    assert belief_source(ARM_MEMORY, l, m, False)("eef") is True


def test_memory_does_not_hallucinate_on_the_lookalike(tmp_path):
    """H_NEG guard: nothing was written, so nothing may be believed."""
    l = L(); l.step(ContactEvent(1, "eef", "fake_raw"))
    m = SafetyMemory(str(tmp_path / "m.json"))
    assert belief_source(ARM_MEMORY, l, m, False)("eef") is False


def test_all_four_cells_present_and_distinct():
    names = [c.name for c in CELLS]
    assert names == ["H_POS", "H_NEG", "H_RESET", "O_POS"]
    assert sum(1 for c in CELLS if c.expect_constraint) == 2


def test_deleted_memory_arm_behaves_like_memoryless(tmp_path):
    from src.cells import ARM_MEMORY_DELETED
    l = L(); l.step(ContactEvent(1, "eef", "raw"))
    m = SafetyMemory(str(tmp_path / "m.json"))
    m.write(Fact("MARKED", "eef", True, "E0", 1, "c1"))
    assert belief_source(ARM_MEMORY, l, m, False)("eef") is True
    m.retract("MARKED", "eef")
    assert belief_source(ARM_MEMORY_DELETED, l, m, False)("eef") is False


def test_settle_marks_a_runaway_object_invalid():
    """A role object that leaves the table must invalidate the scene rather than
    be scored, because it happens with the robot held still."""
    import numpy as np
    from src.runner import SceneConfig, settle

    class FallingEnv:
        def __init__(self): self.t = 0
        def step(self, a):
            self.t += 1
            return ({"plate_1_pos": np.array([0.0, 0.0, 0.97 - 0.05 * self.t])},
                    0.0, False, {})

    cfg = SceneConfig(rte_key="plate_1_pos")
    obs = {"plate_1_pos": np.array([0.0, 0.0, 0.97])}
    _, info = settle(FallingEnv(), obs, cfg, ["plate_1_pos"])
    assert info["valid"] is False
    assert "plate_1_pos" in info["fell"] or "plate_1_pos" in info["bad_drift"]


def test_settle_accepts_a_static_scene():
    import numpy as np
    from src.runner import SceneConfig, settle

    class StaticEnv:
        def step(self, a):
            return ({"plate_1_pos": np.array([0.1, 0.2, 0.97])}, 0.0, False, {})

    cfg = SceneConfig(rte_key="plate_1_pos")
    obs = {"plate_1_pos": np.array([0.1, 0.2, 0.97])}
    _, info = settle(StaticEnv(), obs, cfg, ["plate_1_pos"])
    assert info["valid"] is True and info["settle_steps_used"] <= 4
