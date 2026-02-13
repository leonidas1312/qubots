"""Example optimizer: deterministic hill climbing with restarts."""

import time
from typing import Any

from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class HillClimbOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.steps = 200
        self.restarts = 3

    @staticmethod
    def _flip(solution: list[int], index: int) -> list[int]:
        candidate = list(solution)
        candidate[index] = 1 - int(candidate[index])
        return candidate

    def optimize(self, problem: Any) -> Result:
        start = time.perf_counter()

        best_solution = None
        best_value = float("inf")
        trace: list[float] = []

        for _ in range(int(self.restarts)):
            current = list(problem.random_solution())
            current_value = float(problem.evaluate(current))

            if current_value < best_value:
                best_value = current_value
                best_solution = current

            for _ in range(int(self.steps)):
                step_best_solution = current
                step_best_value = current_value

                for idx in range(len(current)):
                    candidate = self._flip(current, idx)
                    candidate_value = float(problem.evaluate(candidate))
                    if candidate_value < step_best_value:
                        step_best_value = candidate_value
                        step_best_solution = candidate

                if step_best_value < current_value:
                    current = step_best_solution
                    current_value = step_best_value
                    if current_value < best_value:
                        best_value = current_value
                        best_solution = current
                else:
                    break

                trace.append(best_value)

        runtime = time.perf_counter() - start
        return Result(
            best_value=float(best_value),
            best_solution=best_solution,
            runtime_seconds=float(runtime),
            status="ok",
            trace=trace,
            metadata={"steps": int(self.steps), "restarts": int(self.restarts)},
        )
