# File: rastion_core/problems/knapsack.py

import numpy as np
from rastion_core.base_problem import BaseProblem

def setup_problem(num_items, capacity, min_value=1, max_value=20, min_weight=1, max_weight=10):
    """
    Set up a random instance of the knapsack problem.
    Returns lists of item identifiers, values, and weights.
    """
    np.random.seed(123)
    items = [f'item{i}' for i in range(num_items)]
    values = {item: np.random.randint(min_value, max_value+1) for item in items}
    weights = {item: np.random.randint(min_weight, max_weight+1) for item in items}
    return items, values, weights, capacity

def create_qubo_matrix(items, values, weights, capacity, penalty_factor=50):
    """
    Build the QUBO matrix for the knapsack problem.
    
    Q(x) = - sum_i v_i x_i + P*(sum_i w_i x_i - C)^2
         = - sum_i v_i x_i + P*(sum_i w_i^2 x_i + 2*sum_{i<j} w_i w_j x_i x_j - 2C sum_i w_i x_i + C^2)
    The constant term (P * C^2) can be dropped.
    """
    num_items = len(items)
    Q = np.zeros((num_items, num_items))
    # Map item to index for easier QUBO building.
    item_index = {item: idx for idx, item in enumerate(items)}
    
    # Linear terms: from the -value and the squared penalty expansion.
    for item in items:
        idx = item_index[item]
        # From penalty: P * (w_i^2 - 2C * w_i)
        Q[idx, idx] += penalty_factor * (weights[item]**2 - 2 * capacity * weights[item])
        # From objective: -value
        Q[idx, idx] += -values[item]
    
    # Quadratic terms: penalty cross terms.
    for i in range(num_items):
        for j in range(i+1, num_items):
            item_i = items[i]
            item_j = items[j]
            Q[i, j] += 2 * penalty_factor * (weights[item_i] * weights[item_j])
            Q[j, i] = Q[i, j]  # Ensure symmetry
            
    return Q

class KnapsackProblem(BaseProblem):
    """
    A QUBO formulation of the 0-1 knapsack problem.
    This class sets up a random knapsack instance and builds the corresponding QUBO matrix.
    """
    def __init__(self, num_items=8, capacity=20, min_value=1, max_value=20, min_weight=1, max_weight=10, penalty_factor=50):
        self.items, self.values, self.weights, self.capacity = setup_problem(num_items, capacity, min_value, max_value, min_weight, max_weight)
        self.num_items = num_items
        self.QUBO_matrix = create_qubo_matrix(self.items, self.values, self.weights, self.capacity, penalty_factor)
        self.qubo_constant = penalty_factor * (self.capacity ** 2)  # constant term can be dropped if desired.

    def evaluate_solution(self, solution) -> float:
        # Evaluate the QUBO energy.
        sol = np.array(solution)
        return float(sol.T @ self.QUBO_matrix @ sol + self.qubo_constant)

    def random_solution(self):
        # Return a random binary string of length equal to num_items.
        return np.random.randint(0, 2, self.num_items).tolist()

    def get_qubo(self):
        """
        Return the QUBO matrix and constant.
        """
        return self.QUBO_matrix, self.qubo_constant
