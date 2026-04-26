import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    step: int
    uncertainty: Dict[str, Any]
    # Index of this policy-query within the episode (0-based).
    query_step_idx: int = 0
    # Number of env steps executed since the previous query (open-loop chunk size).
    steps_executed: int = 0


@dataclass
class EpisodeRecord:
    episode_idx: int
    task_id: int
    task_description: str
    success: bool
    collide: bool
    steps: int
    step_records: List[StepRecord] = field(default_factory=list)


class UncertaintyLogger:
    """Accumulates per-step uncertainty dicts and flushes to JSON on save()."""

    def __init__(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self._episodes: List[EpisodeRecord] = []
        self._current: Optional[EpisodeRecord] = None
        self._query_step_idx: int = 0
        self._last_query_env_step: int = 0

    def begin_episode(self, episode_idx: int, task_id: int, task_description: str) -> None:
        self._current = EpisodeRecord(
            episode_idx=episode_idx,
            task_id=task_id,
            task_description=task_description,
            success=False,
            collide=False,
            steps=0,
        )
        self._query_step_idx = 0
        self._last_query_env_step = 0

    def log_step(self, step: int, uncertainty: Dict[str, Any]) -> None:
        if self._current is None:
            raise RuntimeError("Call begin_episode() before log_step()")
        steps_executed = step - self._last_query_env_step
        self._current.step_records.append(StepRecord(
            step=step,
            uncertainty=uncertainty,
            query_step_idx=self._query_step_idx,
            steps_executed=steps_executed,
        ))
        self._query_step_idx += 1
        self._last_query_env_step = step

    def end_episode(self, success: bool, collide: bool, steps: int) -> None:
        if self._current is None:
            return
        self._current.success = success
        self._current.collide = collide
        self._current.steps = steps
        self._episodes.append(self._current)
        self._current = None

    def save(self, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump([asdict(e) for e in self._episodes], f, indent=2)
        return path
