import numpy as np
from qubots.base_problem import BaseProblem

def setup_problem(num_vertices, num_colors, edge_probability=0.3):
    """
    Set up a random graph for the graph coloring problem.
    Returns vertex labels and a list of edges (as tuples of vertex indices).
    """
    np.random.seed(123)
    vertices = [f'v{i}' for i in range(num_vertices)]
    edges = []
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            if np.random.rand() < edge_probability:
                edges.append((i, j))
    return vertices, edges, num_colors

def create_qubo_matrix(vertices, edges, num_colors, penalty=10):
    """
    Build the QUBO matrix for the graph coloring problem.
    
    Variables: x_{i,c} for each vertex i and color c.
    Total number of variables: num_vertices * num_colors.
    
    The QUBO objective includes:
      1. Assignment Constraint: Each vertex gets exactly one color.
         For vertex i: penalty * (sum_{c} x_{i,c} - 1)^2.
      2. Adjacency Constraint: For each edge (i,j) and each color c, add penalty * x_{i,c} * x_{j,c}.
    """
    num_vertices = len(vertices)
    num_vars = num_vertices * num_colors
    Q = np.zeros((num_vars, num_vars))
    
    def idx(i, c):
        return i * num_colors + c
    
    # Assignment Constraint for each vertex.
    # (sum_{c} x_{i,c} - 1)^2 = sum_{c} x_{i,c}^2 + 2 * sum_{c < c'} x_{i,c} x_{i,c'} - 2 * sum_{c} x_{i,c} + 1.
    # Since x_{i,c}^2 = x_{i,c} for binary variables.
    for i in range(num_vertices):
        for c in range(num_colors):
            index = idx(i, c)
            Q[index, index] += penalty  # from x_{i,c}
            # Linear term: -2 penalty will be added in the diagonal.
            Q[index, index] += -2 * penalty
            for c2 in range(c+1, num_colors):
                index2 = idx(i, c2)
                Q[index, index2] += 2 * penalty
                Q[index2, index] = Q[index, index2]
    
    # Adjacency Constraint for each edge.
    # For each edge (i,j) and each color c, add penalty * x_{i,c} x_{j,c}.
    for (i, j) in edges:
        for c in range(num_colors):
            index_i = idx(i, c)
            index_j = idx(j, c)
            Q[index_i, index_j] += penalty
            Q[index_j, index_i] = Q[index_i, index_j]
    
    # (A constant offset is dropped as it does not affect optimization.)
    return Q

class GraphColoringProblem(BaseProblem):
    """
    A QUBO formulation of the graph coloring problem.
    This class sets up a random graph and builds the corresponding QUBO matrix.
    """
    def __init__(self, num_vertices=5, num_colors=3, edge_probability=0.3, penalty=10):
        self.vertices, self.edges, self.num_colors = setup_problem(num_vertices, num_colors, edge_probability)
        self.num_vertices = num_vertices
        self.QUBO_matrix = create_qubo_matrix(self.vertices, self.edges, num_colors, penalty)
        self.qubo_constant = 0

    def evaluate_solution(self, solution) -> float:
        # Evaluate the QUBO energy.
        sol = np.array(solution)
        return float(sol.T @ self.QUBO_matrix @ sol + self.qubo_constant)

    def random_solution(self):
        # Return a random binary string of length num_vertices * num_colors.
        size = self.num_vertices * self.num_colors
        return np.random.randint(0, 2, size).tolist()

    def get_qubo(self):
        """
        Return the QUBO matrix and constant.
        """
        return self.QUBO_matrix, self.qubo_constant
