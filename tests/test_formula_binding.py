"""S1 / S5 / S6: variable binding and the GRIPPING threshold.

Three defects that together made the Level-2 semantic rule layer inert:

- S1 `filter.py:1203` substituted variables with a bare `str.replace`, so the
  single letter "A" also replaced the "A" inside "AND", corrupting every seed
  formula. `parse_formula` then raised and a bare `except Exception: continue`
  swallowed it.
- S6 `filter.py:1259,1261` guarded on `"A" in formula`, a substring test that is
  true for every formula containing AND / ABOVE / IS_SPILLABLE.
- S5 `primitives.py:172` compared `gripper_width` (metres, Panda range 0-0.04)
  against 0.5, so GRIPPING was always true and degenerated to NEAR.
"""
import pytest

from fol_safety_filter.filter import _substitute_vars, _formula_uses_var
from fol_safety_filter.kb import parse_formula


# --- S1: substitution must not corrupt operators -------------------------

def test_substitution_does_not_corrupt_AND():
    f = "IS_LIT(A) AND IS_FLAMMABLE(B) AND NEAR(eef, A, 0.30)"
    out = _substitute_vars(f, {"A": "candle", "B": "napkin"})
    assert "candleND" not in out
    assert out == "IS_LIT(candle) AND IS_FLAMMABLE(napkin) AND NEAR(eef, candle, 0.30)"
    parse_formula(out)  # must not raise


def test_substitution_does_not_corrupt_predicate_names_containing_the_var():
    # "B" occurs inside IS_SPILLABLE, ABOVE, IS_ABSORBENT
    out = _substitute_vars("IS_SPILLABLE(A) AND ABOVE(eef, B)", {"A": "pot", "B": "bowl"})
    assert out == "IS_SPILLABLE(pot) AND ABOVE(eef, bowl)"
    parse_formula(out)


def test_substitution_is_not_order_dependent():
    """Substituting A must not corrupt a later B substitution, or vice versa."""
    f = "SUPPORTS(A, B) AND NEAR(eef, A, 0.2)"
    assert _substitute_vars(f, {"A": "tray", "B": "cup"}) == \
        _substitute_vars(f, {"B": "cup", "A": "tray"})


def test_every_composed_rule_seed_parses_after_substitution():
    """The regression that mattered: all 7 seeds were silently discarded."""
    from fol_safety_filter.primitives import COMPOSED_RULE_SEEDS
    assert COMPOSED_RULE_SEEDS, "no seeds defined"
    for seed in COMPOSED_RULE_SEEDS:
        out = _substitute_vars(seed["formula"], {"A": "objA", "B": "objB"})
        parse_formula(out)  # must not raise for any seed


# --- S6: variable-usage tests must be word-boundary ----------------------

def test_var_usage_is_word_boundary_not_substring():
    assert _formula_uses_var("IS_SPILLABLE(A) AND TILTED(eef)", "A") is True
    assert _formula_uses_var("IS_SPILLABLE(A) AND TILTED(eef)", "B") is False
    assert _formula_uses_var("ABOVE(eef, B)", "A") is False
    assert _formula_uses_var("ABOVE(eef, B)", "B") is True


def test_var_usage_false_for_formula_with_only_operators():
    assert _formula_uses_var("NEAR(eef, obj, 0.2) AND TILTED(eef)", "A") is False


# --- S5: GRIPPING threshold is in metres ---------------------------------

def test_gripping_threshold_is_metric():
    from fol_safety_filter.primitives import GRIPPER_CLOSED_M
    # Panda gripper_width range is roughly 0-0.04 m.
    assert 0.0 < GRIPPER_CLOSED_M < 0.04, "threshold must be inside the metre range"


def test_gripping_distinguishes_open_from_closed():
    from fol_safety_filter import primitives as P

    class _S:
        gripper_width = 0.0
        def __init__(self, w):
            self.gripper_width = w

    orig_near = P.NEAR
    P.NEAR = lambda state, subj, obj, proximity: True   # isolate the width test
    try:
        assert P.GRIPPING(_S(0.039), "obj") is False   # open  -> not gripping
        assert P.GRIPPING(_S(0.005), "obj") is True    # closed -> gripping
    finally:
        P.NEAR = orig_near
