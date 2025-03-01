from collections import defaultdict
from qubots.base_problem import BaseProblem
from qubots.formulations.qubo_tsp_problem import QUBOTSPProblem

class IsingTSPProblem(BaseProblem):
    def __init__(self, tsp_problem):
        self.tsp_problem = tsp_problem
        self.qubo_problem = QUBOTSPProblem(tsp_problem)  # Reuse QUBO
        self.h, self.J, self.offset, self.edges = self._convert_to_ising()

    def _convert_to_ising(self):
        Q = self.qubo_problem.qubo
        offset = 0
        h = defaultdict(float)
        J = defaultdict(float)
        edges = []

        max_index = max(max(i, j) for i, j in Q.keys())
        n_qubits = max_index + 1

        for i in range(n_qubits):
            h[i] -= Q.get((i, i), 0) / 2.0
            offset += Q.get((i, i), 0) / 2.0
            for j in range(i+1, n_qubits):
                if Q.get((i, j), 0) != 0:
                    edges.append((i, j))
                J[(i, j)] += Q.get((i, j), 0) / 4.0
                h[i] -= Q.get((i, j), 0) / 4.0
                h[j] -= Q.get((i, j), 0) / 4.0
                offset += Q.get((i, j), 0) / 4.0

        return h, J, offset, edges

    def evaluate_solution(self, spin_solution):
        energy = self.offset
        for i, hi in self.h.items():
            energy += hi * spin_solution[i]
        for (i, j), Jij in self.J.items():
            energy += Jij * spin_solution[i] * spin_solution[j]
        return energy