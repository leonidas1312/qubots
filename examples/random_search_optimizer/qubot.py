"""Example optimizer: random search."""

import time
from typing import Any

from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.iterations = 200

    def optimize(self, problem: Any) -> Result:
        start = time.perf_counter()
        best_value = float("inf")
        best_solution = None
        trace: list[float] = []

        for _ in range(int(self.iterations)):
            candidate = problem.random_solution()
            value = float(problem.evaluate(candidate))
            if value < best_value:
                best_value = value
                best_solution = candidate
            trace.append(best_value)

        runtime = time.perf_counter() - start
        return Result(
            best_value=float(best_value),
            best_solution=best_solution,
            runtime_seconds=float(runtime),
            trace=trace,
            metadata={"iterations": int(self.iterations)},
        )
