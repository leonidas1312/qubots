"""Pipeline API for running problem+optimizer pairs."""

from pathlib import Path
from typing import Any

from qubots.auto.auto_optimizer import AutoOptimizer
from qubots.auto.auto_problem import AutoProblem
from qubots.core.types import Result


class Pipeline:
    def __init__(
        self,
        problem: str | Path,
        optimizer: str | Path,
        trained: str | Path | None = None,
    ) -> None:
        self.problem_repo = problem
        self.optimizer_repo = optimizer
        self.trained = Path(trained) if trained is not None else None

    def __call__(
        self,
        *,
        problem_params: dict[str, Any] | None = None,
        optimizer_params: dict[str, Any] | None = None,
    ) -> Result:
        problem = AutoProblem.from_repo(self.problem_repo)
        optimizer = AutoOptimizer.from_repo(self.optimizer_repo)

        if self.trained is not None:
            optimizer.apply_trained(self.trained)

        if problem_params:
            problem.set_parameters(**problem_params)
        if optimizer_params:
            optimizer.set_parameters(**optimizer_params)

        return optimizer.optimize(problem)


def pipeline(
    problem: str | Path,
    optimizer: str | Path,
    trained: str | Path | None = None,
) -> Pipeline:
    return Pipeline(problem=problem, optimizer=optimizer, trained=trained)
