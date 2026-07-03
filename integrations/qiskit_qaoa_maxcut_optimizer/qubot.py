"""Small Qiskit QAOA-style MaxCut optimizer integration.

This is a local statevector demo for small graphs. It does not require Aer or
cloud hardware and deliberately rejects inputs above ``max_qubits``.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class QiskitQAOAMaxCutOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.capabilities = ["blackbox", "graph"]
        self.p = 1
        self.gamma_grid = 9
        self.beta_grid = 9
        self.shots = 512
        self.seed = 0
        self.max_qubits = 10

    @staticmethod
    def _problem_with_edges(problem: Any) -> Any:
        if hasattr(problem, "_load_edges"):
            return problem
        if hasattr(problem, "_problem"):
            loaded = problem._problem()
            if hasattr(loaded, "_load_edges"):
                return loaded
        raise TypeError(
            "QiskitQAOAMaxCutOptimizer requires an edge-list MaxCut problem "
            "created by qubots detect/import, or another problem exposing "
            "_load_edges()."
        )

    @staticmethod
    def _bits(index: int, n_qubits: int) -> list[int]:
        return [(index >> bit) & 1 for bit in range(n_qubits)]

    @staticmethod
    def _expected_value(problem: Any, statevector: Statevector, n_qubits: int) -> float:
        total = 0.0
        for index, amplitude in enumerate(statevector.data):
            probability = float(abs(amplitude) ** 2)
            if probability:
                total += probability * float(problem.evaluate(QiskitQAOAMaxCutOptimizer._bits(index, n_qubits)))
        return float(total)

    @staticmethod
    def _sample_best(
        problem: Any,
        statevector: Statevector,
        n_qubits: int,
        *,
        shots: int,
        seed: int,
    ) -> tuple[list[int], float, float]:
        probabilities = [float(abs(amplitude) ** 2) for amplitude in statevector.data]
        rng = random.Random(seed)
        best_solution: list[int] | None = None
        best_value = float("inf")
        best_probability = 0.0

        for _ in range(max(1, int(shots))):
            pick = rng.random()
            cumulative = 0.0
            chosen = len(probabilities) - 1
            for index, probability in enumerate(probabilities):
                cumulative += probability
                if pick <= cumulative:
                    chosen = index
                    break
            solution = QiskitQAOAMaxCutOptimizer._bits(chosen, n_qubits)
            value = float(problem.evaluate(solution))
            if value < best_value:
                best_solution = solution
                best_value = value
                best_probability = probabilities[chosen]

        return best_solution or [0] * n_qubits, float(best_value), float(best_probability)

    @staticmethod
    def _circuit(
        n_qubits: int,
        edges: list[tuple[int, int, float]],
        *,
        p: int,
        gamma: float,
        beta: float,
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(n_qubits)
        for qubit in range(n_qubits):
            circuit.h(qubit)
        for _ in range(max(1, int(p))):
            for u, v, weight in edges:
                # Up to a global phase, this applies exp(-i gamma * cut_edge).
                circuit.rzz(-float(gamma) * float(weight), int(u), int(v))
            for qubit in range(n_qubits):
                circuit.rx(2.0 * float(beta), qubit)
        return circuit

    def optimize(self, problem: Any) -> Result:
        start = time.perf_counter()
        edge_problem = self._problem_with_edges(problem)
        n_qubits, edges = edge_problem._load_edges()
        n_qubits = int(n_qubits)

        if n_qubits > int(self.max_qubits):
            return Result(
                best_value=float("inf"),
                best_solution=None,
                runtime_seconds=float(time.perf_counter() - start),
                status="unsupported",
                error=(
                    f"Statevector QAOA demo supports at most {int(self.max_qubits)} "
                    f"qubits, got {n_qubits}."
                ),
            )

        gamma_values = [
            math.pi * i / max(1, int(self.gamma_grid) - 1)
            for i in range(max(1, int(self.gamma_grid)))
        ]
        beta_values = [
            (math.pi / 2.0) * i / max(1, int(self.beta_grid) - 1)
            for i in range(max(1, int(self.beta_grid)))
        ]

        best_state: Statevector | None = None
        best_expected = float("inf")
        best_gamma = 0.0
        best_beta = 0.0

        for gamma in gamma_values:
            for beta in beta_values:
                circuit = self._circuit(
                    n_qubits,
                    edges,
                    p=int(self.p),
                    gamma=gamma,
                    beta=beta,
                )
                statevector = Statevector.from_instruction(circuit)
                expected = self._expected_value(edge_problem, statevector, n_qubits)
                if expected < best_expected:
                    best_expected = expected
                    best_state = statevector
                    best_gamma = gamma
                    best_beta = beta

        if best_state is None:
            best_solution = [0] * n_qubits
            best_value = float(edge_problem.evaluate(best_solution))
            best_probability = 1.0
        else:
            best_solution, best_value, best_probability = self._sample_best(
                edge_problem,
                best_state,
                n_qubits,
                shots=int(self.shots),
                seed=int(self.seed),
            )

        return Result(
            best_value=float(best_value),
            best_solution=best_solution,
            runtime_seconds=float(time.perf_counter() - start),
            status="ok",
            metadata={
                "framework": "qiskit",
                "algorithm": "qaoa_grid_statevector",
                "p": int(self.p),
                "n_qubits": n_qubits,
                "edges": len(edges),
                "gamma": float(best_gamma),
                "beta": float(best_beta),
                "expected_value": float(best_expected),
                "sample_probability": float(best_probability),
                "cut_weight": -float(best_value),
                "shots": int(self.shots),
            },
        )
