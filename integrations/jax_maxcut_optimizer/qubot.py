"""JAX-backed MaxCut local-search optimizer integration."""

from __future__ import annotations

import random
import time
from typing import Any

import jax
import jax.numpy as jnp

from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class JAXMaxCutOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.capabilities = ["blackbox", "graph"]
        self.max_passes = 20
        self.seed = 0

    @staticmethod
    def _problem_with_edges(problem: Any) -> Any:
        if hasattr(problem, "_load_edges"):
            return problem
        if hasattr(problem, "_problem"):
            loaded = problem._problem()
            if hasattr(loaded, "_load_edges"):
                return loaded
        raise TypeError(
            "JAXMaxCutOptimizer requires an edge-list MaxCut problem "
            "created by qubots detect/import, or another problem exposing "
            "_load_edges()."
        )

    def optimize(self, problem: Any) -> Result:
        start = time.perf_counter()
        edge_problem = self._problem_with_edges(problem)
        n_nodes, edges = edge_problem._load_edges()
        n_nodes = int(n_nodes)
        if n_nodes == 0:
            return Result(0.0, [], 0.0, metadata={"framework": "jax", "nodes": 0})

        u = jnp.array([int(edge[0]) for edge in edges], dtype=jnp.int32)
        v = jnp.array([int(edge[1]) for edge in edges], dtype=jnp.int32)
        w = jnp.array([float(edge[2]) for edge in edges], dtype=jnp.float32)

        @jax.jit
        def negative_cut(labels: Any) -> Any:
            return -jnp.sum(jnp.where(labels[u] != labels[v], w, 0.0))

        rng = random.Random(int(self.seed))
        labels = jnp.array([rng.randint(0, 1) for _ in range(n_nodes)], dtype=jnp.int32)
        best_value = float(negative_cut(labels))
        trace = [best_value]

        for _ in range(int(self.max_passes)):
            improved = False
            for node in range(n_nodes):
                candidate = labels.at[node].set(1 - labels[node])
                value = float(negative_cut(candidate))
                if value < best_value:
                    labels = candidate
                    best_value = value
                    improved = True
            trace.append(best_value)
            if not improved:
                break

        solution = [int(value) for value in labels.tolist()]
        return Result(
            best_value=float(edge_problem.evaluate(solution)),
            best_solution=solution,
            runtime_seconds=float(time.perf_counter() - start),
            status="ok",
            trace=trace,
            metadata={
                "framework": "jax",
                "nodes": n_nodes,
                "edges": len(edges),
                "cut_weight": -best_value,
                "max_passes": int(self.max_passes),
            },
        )
