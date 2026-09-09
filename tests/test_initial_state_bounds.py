"""L11: initial-state selection must fail loudly, never wrap.

Three drivers indexed `initial_states[episode_idx]` bare (opaque IndexError,
reachable via --episode_offset) and two wrapped with `% len(initial_states)`,
which silently re-runs earlier episodes -- inflating the apparent sample size
with duplicates while the results JSON still reports the requested count.
"""
import pytest

from libsafety_env_utils import select_initial_state


STATES = ["s0", "s1", "s2"]


def test_returns_the_requested_state():
    assert select_initial_state(STATES, 0) == "s0"
    assert select_initial_state(STATES, 2) == "s2"


def test_raises_past_the_end_rather_than_wrapping():
    with pytest.raises(IndexError) as exc:
        select_initial_state(STATES, 3)
    msg = str(exc.value)
    assert "3" in msg and "3 available" in msg
    # The old `% len` behaviour would have silently returned "s0".
    assert "s0" not in msg


def test_raises_on_negative_index():
    """Negative indices would silently select from the end of the list."""
    with pytest.raises(IndexError):
        select_initial_state(STATES, -1)


def test_message_names_the_task_when_given():
    with pytest.raises(IndexError, match=r"task 7"):
        select_initial_state(STATES, 99, task_index=7)


def test_empty_state_list_raises():
    with pytest.raises(IndexError):
        select_initial_state([], 0)
