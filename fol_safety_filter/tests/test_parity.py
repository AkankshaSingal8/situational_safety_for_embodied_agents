"""Bit-exact parity of ``_apply_cbf`` against the committed golden trace.

Every number in ``fol_safety_filter/README.md`` came out of the hardcoded
obstacle loop in ``_apply_cbf``.  Routing those decisions through loaded rule
records is a restructuring only if the emitted actions do not move, so this
test is the acceptance bar for that work rather than a smoke check.

Scope, stated because it is narrower than it looks: the harness builds
``MappedCBFSet`` itself, so ``kb.evaluate`` and ``mapper.map`` are not in the
loop.  This pins the projection and the two clamps **given a fixed constraint
set**; it does not pin which rules fire.  Rule selection is covered by
``test_rule_binding.py``, ``test_effect_dispatch.py`` and
``test_rule_falsifiability.py``.
"""

from __future__ import annotations

import collections
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_TOOLS = os.path.join(_REPO_ROOT, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

GOLDEN = os.path.join(_HERE, "data", "parity_golden.npz")


@pytest.fixture(scope="module")
def golden():
    if not os.path.exists(GOLDEN):
        pytest.skip(f"golden trace missing: {GOLDEN}")
    return np.load(GOLDEN, allow_pickle=True)


@pytest.fixture(scope="module")
def current():
    import fol_parity_trace

    return fol_parity_trace.compute_trace()


def test_scenario_keys_match(golden, current):
    assert [str(k) for k in golden["keys"]] == [str(k) for k in current["keys"]]


def test_u_safe_is_bit_identical(golden, current):
    delta = np.abs(current["u_safe"] - golden["u_safe"])
    max_delta = float(delta.max())
    if max_delta != 0.0:
        row = int(delta.max(axis=1).argmax())
        key = str(current["keys"][row])
        pytest.fail(
            f"max|delta| = {max_delta:.3e}, must be exactly 0.\n"
            f"  worst row: {key}\n"
            f"  golden : {golden['u_safe'][row]}\n"
            f"  current: {current['u_safe'][row]}\n"
            "Do not regenerate the golden trace to make this pass."
        )


def test_golden_trace_is_not_vacuous(golden):
    """A trace where the filter never intervenes would pass trivially."""
    active = np.abs(golden["u_safe"] - golden["u_cmd"]).max(axis=1) > 1e-12
    assert active.mean() > 0.5, (
        f"only {active.mean():.1%} of rows have u_safe != u_cmd"
    )


def test_every_scenario_and_env_combo_contributes(golden):
    """Coverage, not just volume: each branch must actually be exercised."""
    active = np.abs(golden["u_safe"] - golden["u_cmd"]).max(axis=1) > 1e-12
    per_scenario = collections.Counter()
    per_env = collections.Counter()
    for key, is_active in zip(golden["keys"], active):
        env, name, _ = str(key).split("|")
        if is_active:
            per_scenario[name] += 1
            per_env[env] += 1

    import fol_parity_trace

    names = {s["name"] for s in fol_parity_trace.build_scenarios()}
    silent_scenarios = sorted(names - set(per_scenario))
    assert not silent_scenarios, f"scenarios never active: {silent_scenarios}"

    envs = {f"e{e}_r{r}_b{b}" for e, r, b in fol_parity_trace.ENV_COMBOS}
    silent_envs = sorted(envs - set(per_env))
    assert not silent_envs, f"env combos never active: {silent_envs}"


def test_ellipsoid_and_ray_rescale_branches_differ(golden):
    """Guard against an env-var axis that silently selects the same code."""
    active = np.abs(golden["u_safe"] - golden["u_cmd"]).max(axis=1) > 1e-12
    counts = collections.Counter()
    for key, is_active in zip(golden["keys"], active):
        env, _, _ = str(key).split("|")
        if is_active:
            counts[env] += 1
    # With ellipsoids on, ray-rescale takes a different path from slide-cancel;
    # if these ever coincide the matrix has stopped testing anything.
    assert counts["e1_r1_b0"] != counts["e1_r0_b0"]
    assert counts["e1_rtarget_b0"] != counts["e1_r0_b0"]
