"""Example problem: deterministic binary knapsack."""

import random

from qubots.core.problem import BaseProblem


class KnapsackProblem(BaseProblem):
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

    def evaluate(self, solution: list[int]) -> float:
        self._ensure_instance()

        clipped = solution[: len(self._weights)]
        if len(clipped) < len(self._weights):
            clipped = clipped + [0] * (len(self._weights) - len(clipped))

        total_weight = sum(w for w, bit in zip(self._weights, clipped) if bit)
        total_value = sum(v for v, bit in zip(self._values, clipped) if bit)

        overweight = max(0, total_weight - self._capacity)
        penalty = float(100 * overweight)

        # Minimize negative value + overweight penalty.
        return -float(total_value) + penalty

    def random_solution(self) -> list[int]:
        self._ensure_instance()

        # Deterministic random sequence for a fixed problem seed.
        rng = random.Random(int(self.seed) + self._sample_count * 1_000_003)
        self._sample_count += 1
        return [rng.randint(0, 1) for _ in range(len(self._weights))]
