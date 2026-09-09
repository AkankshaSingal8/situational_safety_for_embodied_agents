# epistemic_uncertainty/tests/test_logger.py
import json
import os
import tempfile
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from epistemic_uncertainty.logger import UncertaintyLogger


def test_log_full_episode_and_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        log = UncertaintyLogger(output_dir=tmpdir)
        log.begin_episode(episode_idx=0, task_id=1, task_description="pick up bowl")
        log.log_step(step=0, uncertainty={"mc_dropout": {"variance": 0.05}})
        log.log_step(step=8, uncertainty={"mc_dropout": {"variance": 0.12}})
        log.end_episode(success=True, collide=False, steps=42)

        path = log.save("test_run.json")
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)

        assert len(data) == 1
        ep = data[0]
        assert ep["episode_idx"] == 0
        assert ep["task_id"] == 1
        assert ep["success"] is True
        assert ep["collide"] is False
        assert ep["steps"] == 42
        assert len(ep["step_records"]) == 2
        assert ep["step_records"][0]["step"] == 0
        assert ep["step_records"][1]["uncertainty"]["mc_dropout"]["variance"] == 0.12


def test_multiple_episodes_accumulated():
    with tempfile.TemporaryDirectory() as tmpdir:
        log = UncertaintyLogger(output_dir=tmpdir)
        for i in range(3):
            log.begin_episode(episode_idx=i, task_id=0, task_description="task")
            log.log_step(step=0, uncertainty={})
            log.end_episode(success=(i % 2 == 0), collide=False, steps=10)

        path = log.save("multi.json")
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 3
        assert data[1]["success"] is False


def test_no_begin_episode_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        log = UncertaintyLogger(output_dir=tmpdir)
        import pytest
        with pytest.raises(RuntimeError):
            log.log_step(step=0, uncertainty={})
