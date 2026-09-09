"""Tests for the LIBERO-Safety result aggregator.

Two defects pinned here:

- ETS_median was computed over per-task MEANS repeated `episodes` times, not
  over episodes, so it was the median of task means.
- The expected task count was hardcoded to `set(range(15))`. 15 is the correct
  value per the LIBERO-Safety paper (5 tasks x 3 difficulty levels), but it must
  come from the suite registry rather than a literal.
"""
from aggregate_libsafety_results import episode_ets_values, expected_task_ids


def test_ets_values_come_from_episodes_when_available():
    # task A: 3 episodes of 10; task B: 1 episode of 100.
    # Median over episodes is 10.0; median over task means would be 55.0.
    records = [
        {"ETS_mean": 10.0, "episodes": 3, "ets": [10.0, 10.0, 10.0]},
        {"ETS_mean": 100.0, "episodes": 1, "ets": [100.0]},
    ]
    assert sorted(episode_ets_values(records)) == [10.0, 10.0, 10.0, 100.0]


def test_ets_values_fall_back_to_task_mean_for_legacy_files():
    """Stored results predate per-episode ETS logging and must still aggregate."""
    records = [{"ETS_mean": 12.5, "episodes": 2}]
    assert episode_ets_values(records) == [12.5, 12.5]


def test_ets_values_mixed_legacy_and_new_records():
    records = [
        {"ETS_mean": 10.0, "episodes": 2, "ets": [5.0, 15.0]},
        {"ETS_mean": 20.0, "episodes": 2},
    ]
    assert sorted(episode_ets_values(records)) == [5.0, 15.0, 20.0, 20.0]


def test_expected_task_ids_reads_suite_size_not_a_literal():
    assert expected_task_ids(15) == set(range(15))
    assert expected_task_ids(9) == set(range(9))


def test_expected_task_ids_default_matches_the_paper():
    """LIBERO-Safety: 5 tasks x 3 difficulty levels = 15 tasks per suite."""
    assert expected_task_ids() == set(range(15))
