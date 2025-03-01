import numpy as np
from qubots.base_problem import BaseProblem

class QUBOTSPProblem(BaseProblem):
    def __init__(self, tsp_problem):
        self.tsp_problem = tsp_problem
        self.nb_cities = tsp_problem.nb_cities
        self.dist_matrix = np.array(tsp_problem.dist_matrix)
        self.qubo = self._build_qubo()
        # Count unique binary variable indices
        unique_variables = set()
        for (i, j) in self.qubo.keys():
            unique_variables.add(i)
            unique_variables.add(j)

        print(f"QUBO problem created with {len(unique_variables)} classical variables.")


    def _build_qubo(self):
        Q = {}
        n = self.nb_cities
        max_distance = np.max(self.dist_matrix)
        A = max_distance * n
        B = A  # same penalty for constraints

        def add_to_Q(p, q, value):
            if p > q:
                p, q = q, p
            Q[(p, q)] = Q.get((p, q), 0) + value

        # Objective term
        for t in range(n):
            t_next = (t + 1) % n
            for i in range(n):
                for j in range(n):
                    p = i * n + t
                    q = j * n + t_next
                    add_to_Q(p, q, self.dist_matrix[i, j])

        # Constraint 1: Each city once
        for i in range(n):
            for t1 in range(n):
                p = i * n + t1
                add_to_Q(p, p, -A)
                for t2 in range(t1+1, n):
                    q = i * n + t2
                    add_to_Q(p, q, 2 * A)

        # Constraint 2: Each position once
        for t in range(n):
            for i1 in range(n):
                p = i1 * n + t
                add_to_Q(p, p, -B)
                for i2 in range(i1+1, n):
                    q = i2 * n + t
                    add_to_Q(p, q, 2 * B)

        return Q

    def evaluate_solution(self, binary_solution):
        energy = 0.0
        for (i, j), coeff in self.qubo.items():
            if i == j:
                energy += coeff * binary_solution[i]
            else:
                energy += coeff * binary_solution[i] * binary_solution[j]
        return energy
    
    def decode_solution(self, binary_solution):
        # --- Decode the binary vector into a tour ---
        # Reshape binary_solution into an (n x n) assignment matrix.
        binary_solution = np.array(binary_solution)
        n = self.nb_cities
        X = binary_solution.reshape((n, n))
        tour = []
        for t in range(n):
            col = X[:, t]
            # If exactly one city is assigned at position t, use it; otherwise take argmax.
            if np.sum(col) == 1:
                i = int(np.where(col == 1)[0][0])
            else:
                i = int(np.argmax(col))
            tour.append(i)
        # Optionally, rotate the tour so that city 0 is first.
        if 0 in tour:
            idx = tour.index(0)
            tour = tour[idx:] + tour[:idx]

        return tour
    
    def get_qubo(self):
        return self.qubo