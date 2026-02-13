"""Base interface for optimization problems."""

from abc import ABC, abstractmethod
from typing import Any


class BaseProblem(ABC):
    def __init__(self) -> None:
        self.parameters: dict[str, Any] = {}

    def set_parameters(self, **kwargs: Any) -> None:
        self.parameters.update(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def evaluate(self, solution: Any) -> float:
        raise NotImplementedError

    @abstractmethod
    def random_solution(self) -> Any:
        raise NotImplementedError
