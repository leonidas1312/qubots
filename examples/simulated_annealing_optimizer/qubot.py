"""Example optimizer: deterministic simulated annealing."""

import math
import random
import time
from typing import Any

from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class SimulatedAnnealingOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.steps = 300
        self.t0 = 1.0
        self.cooling = 0.995

    def optimize(self, problem: Any) -> Result:
        start = time.perf_counter()

        base_seed = int(getattr(problem, "seed", 0))
        rng = random.Random(base_seed)

        current = list(problem.random_solution())
        current_value = float(problem.evaluate(current))

        best_solution = list(current)
        best_value = float(current_value)
        trace: list[float] = [best_value]

        temperature = max(float(self.t0), 1e-9)

        for _ in range(int(self.steps)):
            idx = rng.randrange(len(current))
            candidate = list(current)
            candidate[idx] = 1 - int(candidate[idx])
            candidate_value = float(problem.evaluate(candidate))

            delta = candidate_value - current_value
            if delta <= 0:
                accept = True
            else:
                accept_prob = math.exp(-delta / max(temperature, 1e-9))
                accept = rng.random() < accept_prob

            if accept:
                current = candidate
                current_value = candidate_value
                if current_value < best_value:
                    best_value = current_value
                    best_solution = list(current)

            trace.append(best_value)
            temperature *= float(self.cooling)
            temperature = max(temperature, 1e-9)

        runtime = time.perf_counter() - start
        return Result(
            best_value=float(best_value),
            best_solution=best_solution,
            runtime_seconds=float(runtime),
            status="ok",
            trace=trace,
            metadata={
                "steps": int(self.steps),
                "t0": float(self.t0),
                "cooling": float(self.cooling),
            },
        )
