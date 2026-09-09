"""L10: a missing `collision` field must not silently read as "no collision".

`aggregate_table3_replication.py` computed SR/CR with
`r.get("collision", False)`, while `run_libsafety_eval_openvla.py` recorded no
collision signal at all. Every OpenVLA row therefore scored SR == TSR and
CR == 0.0%, printed beside the paper's Table 3 as a collision-aware replication.
"""
import pytest

from aggregate_table3_replication import sr_cr


def test_computes_sr_and_cr_when_the_field_is_present():
    records = [
        {"success": True, "collision": False},
        {"success": True, "collision": True},
        {"success": False, "collision": True},
        {"success": False, "collision": False},
    ]
    sr, cr, n = sr_cr(records)
    assert n == 4
    assert sr == 25.0   # only the success WITHOUT a collision counts
    assert cr == 50.0


def test_raises_when_any_record_lacks_the_field():
    records = [{"success": True, "collision": False}, {"success": True}]
    with pytest.raises(KeyError, match="no 'collision' field"):
        sr_cr(records)


def test_error_names_how_many_and_where():
    records = [{"success": True}, {"success": False}]
    with pytest.raises(KeyError) as exc:
        sr_cr(records)
    assert "2 of 2" in str(exc.value)


def test_empty_input_is_not_an_error():
    assert sr_cr([]) == (None, None, 0)


def test_regression_old_default_would_have_scored_sr_equals_tsr():
    """Pin the defect: with .get(...,False), these score SR=100, CR=0."""
    records = [{"success": True}, {"success": True}]
    old_sr = 100 * sum(1 for r in records if r["success"] and not r.get("collision", False)) / 2
    old_cr = 100 * sum(1 for r in records if r.get("collision", False)) / 2
    assert (old_sr, old_cr) == (100.0, 0.0)
    with pytest.raises(KeyError):
        sr_cr(records)
