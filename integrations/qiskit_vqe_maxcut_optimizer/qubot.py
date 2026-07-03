"""Small Qiskit VQE-style MaxCut optimizer integration."""

from __future__ import annotations

import math
import random
import time
from typing import Any

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class QiskitVQEMaxCutOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.capabilities = ["blackbox", "graph"]
        self.trials = 64
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
            "QiskitVQEMaxCutOptimizer requires an edge-list MaxCut problem "
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
                total += probability * float(problem.evaluate(QiskitVQEMaxCutOptimizer._bits(index, n_qubits)))
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
            solution = QiskitVQEMaxCutOptimizer._bits(chosen, n_qubits)
            value = float(problem.evaluate(solution))
            if value < best_value:
                best_solution = solution
                best_value = value
                best_probability = probabilities[chosen]

        return best_solution or [0] * n_qubits, float(best_value), float(best_probability)

    @staticmethod
    def _ansatz(n_qubits: int, parameters: list[float]) -> QuantumCircuit:
        circuit = QuantumCircuit(n_qubits)
        for qubit in range(n_qubits):
            circuit.ry(float(parameters[qubit]), qubit)
        for qubit in range(n_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        for qubit in range(n_qubits):
            circuit.rz(float(parameters[n_qubits + qubit]), qubit)
            circuit.ry(float(parameters[2 * n_qubits + qubit]), qubit)
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
                    f"Statevector VQE demo supports at most {int(self.max_qubits)} "
                    f"qubits, got {n_qubits}."
                ),
            )

        rng = random.Random(int(self.seed))
        best_state: Statevector | None = None
        best_expected = float("inf")
        best_parameters: list[float] = []
        trace: list[float] = []

        for _ in range(max(1, int(self.trials))):
            parameters = [rng.uniform(0.0, 2.0 * math.pi) for _ in range(3 * n_qubits)]
            circuit = self._ansatz(n_qubits, parameters)
            statevector = Statevector.from_instruction(circuit)
            expected = self._expected_value(edge_problem, statevector, n_qubits)
            if expected < best_expected:
                best_state = statevector
                best_expected = expected
                best_parameters = list(parameters)
            trace.append(float(best_expected))

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
            trace=trace,
            metadata={
                "framework": "qiskit",
                "algorithm": "vqe_random_statevector",
                "n_qubits": n_qubits,
                "edges": len(edges),
                "trials": int(self.trials),
                "expected_value": float(best_expected),
                "sample_probability": float(best_probability),
                "cut_weight": -float(best_value),
                "parameter_count": len(best_parameters),
                "shots": int(self.shots),
            },
        )
