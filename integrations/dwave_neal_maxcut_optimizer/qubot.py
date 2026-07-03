"""D-Wave dimod/neal MaxCut optimizer integration."""

from __future__ import annotations

import time
from typing import Any

import dimod
import neal

from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class DWaveNealMaxCutOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.capabilities = ["blackbox", "graph"]
        self.num_reads = 50
        self.sweeps = 100
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
            "DWaveNealMaxCutOptimizer requires an edge-list MaxCut problem "
            "created by qubots detect/import, or another problem exposing "
            "_load_edges()."
        )

    def optimize(self, problem: Any) -> Result:
        start = time.perf_counter()
        edge_problem = self._problem_with_edges(problem)
        n_nodes, edges = edge_problem._load_edges()

        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
        for node in range(int(n_nodes)):
            bqm.add_variable(node, 0.0)
        for u, v, weight in edges:
            weight = float(weight)
            bqm.add_linear(int(u), -weight)
            bqm.add_linear(int(v), -weight)
            bqm.add_quadratic(int(u), int(v), 2.0 * weight)

        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(
            bqm,
            num_reads=int(self.num_reads),
            sweeps=int(self.sweeps),
            seed=int(self.seed),
        )
        sample = sampleset.first.sample
        solution = [int(sample.get(node, 0)) for node in range(int(n_nodes))]
        best_value = float(edge_problem.evaluate(solution))

        return Result(
            best_value=best_value,
            best_solution=solution,
            runtime_seconds=float(time.perf_counter() - start),
            status="ok",
            metadata={
                "framework": "dwave-neal",
                "model": "dimod.BinaryQuadraticModel",
                "nodes": int(n_nodes),
                "edges": len(edges),
                "cut_weight": -best_value,
                "num_reads": int(self.num_reads),
                "sweeps": int(self.sweeps),
            },
        )
