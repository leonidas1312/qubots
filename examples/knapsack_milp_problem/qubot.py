"""Binary knapsack expressed as a MILP and as a blackbox.

This problem exposes ``as_milp()`` so structured solvers like HiGHS can solve
it exactly, while also implementing ``evaluate()`` / ``random_solution()`` so
metaheuristic optimizers (random search, hill climb, simulated annealing) can
attack the same instance.
"""

from __future__ import annotations

import random

from qubots.core.milp import MILPModel
from qubots.core.problem import BaseProblem


class KnapsackMILPProblem(BaseProblem):
    def __init__(self) -> None:
        super().__init__()
        self.n_items = 30
        self.capacity_ratio = 0.4
        self.seed = 0
        self._instance_key: tuple[int, float, int] | None = None
        self._weights: list[int] = []
        self._values: list[int] = []
        self._capacity = 0
        self._sample_count = 0

    def _ensure_instance(self) -> None:
        key = (int(self.n_items), float(self.capacity_ratio), int(self.seed))
        if self._instance_key == key:
            return

        n_items, capacity_ratio, seed = key
        rng = random.Random(seed)
        self._weights = [rng.randint(1, 20) for _ in range(n_items)]
        self._values = [rng.randint(1, 30) for _ in range(n_items)]
        self._capacity = max(1, int(capacity_ratio * sum(self._weights)))
        self._instance_key = key
        self._sample_count = 0

    def as_milp(self) -> MILPModel:
        self._ensure_instance()
        n = len(self._weights)
        return MILPModel(
            sense="max",
            c=[float(v) for v in self._values],
            var_names=[f"take_{i}" for i in range(n)],
            integrality=[True] * n,
            lb=[0.0] * n,
            ub=[1.0] * n,
            A_ub=[[float(w) for w in self._weights]],
            b_ub=[float(self._capacity)],
            constraint_names=["capacity"],
        )

    def evaluate(self, solution: list[int]) -> float:
        self._ensure_instance()
        n = len(self._weights)
        clipped = list(solution[:n]) + [0] * max(0, n - len(solution))

        total_weight = sum(w for w, bit in zip(self._weights, clipped) if bit)
        total_value = sum(v for v, bit in zip(self._values, clipped) if bit)
        overweight = max(0, total_weight - self._capacity)
        penalty = float(100 * overweight)
        return -float(total_value) + penalty

    def random_solution(self) -> list[int]:
        self._ensure_instance()
        rng = random.Random(int(self.seed) + self._sample_count * 1_000_003)
        self._sample_count += 1
        return [rng.randint(0, 1) for _ in range(len(self._weights))]
