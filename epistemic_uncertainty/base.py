from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class UncertaintyEstimate:
    method: str
    step: int
    values: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"method": self.method, "step": self.step, "values": self.values}


class BaseUncertaintyEstimator(ABC):
    """Abstract base for all uncertainty estimators."""

    @abstractmethod
    def estimate(self, observation: Dict[str, Any], step: int) -> UncertaintyEstimate:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...
