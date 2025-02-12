import numpy as np
from itertools import product
from qubots.base_optimizer import BaseOptimizer

class ExhaustiveSearch(BaseOptimizer):
    """
    An exhaustive search optimizer for QUBO problems.
    
    This solver evaluates every possible binary assignment and returns the one
    with the lowest cost. Note that this method is only feasible for very small
    problems (e.g., n < 20).
    """
    def __init__(self):
        pass

    def optimize(self, problem, **kwargs):
        """
        Run an exhaustive search on the given QUBO problem.
        
        The problem instance is expected to implement:
          - evaluate_solution(solution): returns a numerical cost.
          - get_qubo(): returns a tuple (QUBO_matrix, qubo_constant) if needed.
        
        :param problem: The QUBO problem instance.
        :return: A tuple (best_solution, best_cost).
        """
        # Get QUBO matrix and constant (if needed)
        QUBO_matrix, qubo_constant = problem.get_qubo()
        n = QUBO_matrix.shape[0]  # Dimension of the solution vector
        
        best_solution = None
        best_cost = float('inf')

        # Generate all possible binary solutions (each solution is a tuple of 0's and 1's)
        for bits in product([0, 1], repeat=n):
            # Convert tuple to list (or numpy array) as required by evaluate_solution
            solution = list(bits)
            cost = problem.evaluate_solution(solution)
            if cost < best_cost:
                best_cost = cost
                best_solution = solution

        return best_solution, best_cost