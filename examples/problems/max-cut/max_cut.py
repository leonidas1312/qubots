import numpy as np
from qubots.base_problem import BaseProblem

def setup_problem(num_nodes, edge_probability=0.5, min_weight=1, max_weight=10):
    """
    Set up a random weighted graph for the MaxCut problem.
    Returns the list of nodes and an edge weight dictionary.
    """
    np.random.seed(123)
    nodes = [f'node{i}' for i in range(num_nodes)]
    # Create a symmetric weight matrix in dictionary form.
    weights = {}
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if np.random.rand() < edge_probability:
                weight = np.random.randint(min_weight, max_weight+1)
                weights[(i, j)] = weight
                weights[(j, i)] = weight
            else:
                weights[(i, j)] = 0
                weights[(j, i)] = 0
    return nodes, weights

def create_qubo_matrix(num_nodes, weights):
    """
    Build the QUBO matrix for the MaxCut problem.
    
    The cut value (to be maximized) is:
      C(x) = sum_{i<j} w_{ij}(x_i + x_j - 2x_i x_j)
    
    We convert this into a minimization QUBO by minimizing:
      Q(x) = -C(x)
    """
    # The number of variables equals the number of nodes.
    Q = np.zeros((num_nodes, num_nodes))
    
    # Build QUBO from -C(x) = - sum_{i<j} w_{ij}(x_i + x_j) + 2 sum_{i<j} w_{ij} x_i x_j.
    # First, add off-diagonal quadratic terms.
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            weight = weights.get((i, j), 0)
            Q[i, j] += 2 * weight
            Q[j, i] = Q[i, j]  # Ensure symmetry
    
    # Now, add the linear terms to the diagonal.
    for i in range(num_nodes):
        # Sum the weights for all edges incident to node i.
        incident_weight = sum(weights.get((i, j), 0) for j in range(num_nodes) if j != i)
        Q[i, i] += -incident_weight
    
    return Q

class MaxCutProblem(BaseProblem):
    """
    A QUBO formulation of the MaxCut problem.
    This class sets up a random graph and builds the corresponding QUBO matrix.
    """
    def __init__(self, num_nodes=6, edge_probability=0.5, min_weight=1, max_weight=10):
        self.nodes, self.weights = setup_problem(num_nodes, edge_probability, min_weight, max_weight)
        self.num_nodes = num_nodes
        self.QUBO_matrix = create_qubo_matrix(num_nodes, self.weights)
        # A constant offset can be added if needed (here, zero).
        self.qubo_constant = 0

    def evaluate_solution(self, solution) -> float:
        # Evaluate the QUBO energy: solution^T Q solution + constant.
        sol = np.array(solution)
        return float(sol.T @ self.QUBO_matrix @ sol + self.qubo_constant)

    def random_solution(self):
        # Return a random binary string of length equal to num_nodes.
        return np.random.randint(0, 2, self.num_nodes).tolist()

    def get_qubo(self):
        """
        Return the QUBO matrix and constant.
        """
        return self.QUBO_matrix, self.qubo_constant
