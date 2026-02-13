"""Example problem: deterministic Erdos-Renyi MaxCut."""

import random

from qubots.core.problem import BaseProblem


class MaxCutProblem(BaseProblem):
    def __init__(self) -> None:
        super().__init__()
        self.n_nodes = 20
        self.edge_prob = 0.2
        self.seed = 0
        self._instance_key: tuple[int, float, int] | None = None
        self._edges: list[tuple[int, int]] = []
        self._sample_count = 0

    def _ensure_instance(self) -> None:
        key = (int(self.n_nodes), float(self.edge_prob), int(self.seed))
        if self._instance_key == key:
            return

        n_nodes, edge_prob, seed = key
        rng = random.Random(seed)
        edges: list[tuple[int, int]] = []

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < edge_prob:
                    edges.append((i, j))

        self._edges = edges
        self._instance_key = key
        self._sample_count = 0

    def evaluate(self, solution: list[int]) -> float:
        self._ensure_instance()

        n_nodes = int(self.n_nodes)
        clipped = solution[:n_nodes]
        if len(clipped) < n_nodes:
            clipped = clipped + [0] * (n_nodes - len(clipped))

        cut_size = 0
        for i, j in self._edges:
            if clipped[i] != clipped[j]:
                cut_size += 1

        # Minimize negative cut size.
        return -float(cut_size)

    def random_solution(self) -> list[int]:
        self._ensure_instance()

        rng = random.Random(int(self.seed) + self._sample_count * 1_000_003)
        self._sample_count += 1
        return [rng.randint(0, 1) for _ in range(int(self.n_nodes))]
