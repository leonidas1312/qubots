"""Example problem: OneMax."""

import random

from qubots.core.problem import BaseProblem


class OneMaxProblem(BaseProblem):
    def __init__(self) -> None:
        super().__init__()
        self.n_bits = 32

    def evaluate(self, solution: list[int]) -> float:
        # Minimize the negative count of ones, equivalent to maximizing ones.
        return -float(sum(solution))

    def random_solution(self) -> list[int]:
        return [random.randint(0, 1) for _ in range(int(self.n_bits))]
