import pytest
from vlm_prompt_runner.run_majority_vote_experiment import majority_vote, merge_votes


def test_majority_vote_clear_winner():
    votes = ["cup_1", "cup_1", "cup_1", "moka_pot", "cup_1"]
    assert majority_vote(votes) == "cup_1"


def test_majority_vote_tie_returns_first_encountered():
    votes = ["cup_1", "moka_pot", "cup_1", "moka_pot"]
    result = majority_vote(votes)
    assert result in ("cup_1", "moka_pot")


def test_majority_vote_single():
    assert majority_vote(["cup_1"]) == "cup_1"


def test_majority_vote_all_different():
    votes = ["a", "b", "c"]
    assert majority_vote(votes) in votes


def test_majority_vote_empty_raises():
    with pytest.raises(ValueError):
        majority_vote([])


def test_merge_votes_structure():
    votes = ["cup_1", "cup_1", "moka_pot"]
    raw_responses = ["raw1", "raw2", "raw3"]
    meta = {"task_description": "pick cup", "ep_dir": "/some/ep"}
    result = merge_votes(votes, raw_responses, meta)
    assert result["object"] == "cup_1"
    assert result["vote_counts"] == {"cup_1": 2, "moka_pot": 1}
    assert result["all_votes"] == votes
    assert result["_raw_responses"] == raw_responses
    assert result["_meta"] == meta


def test_merge_votes_none_filtered():
    votes = [None, "cup_1", None, "cup_1", None]
    raw_responses = ["r1", "r2", "r3", "r4", "r5"]
    meta = {}
    result = merge_votes(votes, raw_responses, meta)
    assert result["object"] == "cup_1"
