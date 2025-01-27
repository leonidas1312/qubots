import random
from rastion_core.base_problem import BaseProblem

class TSPProblem(BaseProblem):
    """
    Traveling Salesman Problem:
    - We have a distance matrix of size n x n (n cities).
    - A solution is a permutation of [0..n-1] indicating the order of cities visited.
    """

    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)

    def evaluate_solution(self, tour) -> float:
        """
        Compute the total distance of the TSP route (tour).
        The tour is assumed to be a list of city indices in visiting order.
        """
        total_dist = 0.0
        for i in range(len(tour) - 1):
            total_dist += self.distance_matrix[tour[i]][tour[i+1]]
        # Add distance from the last city back to the first (typical TSP)
        total_dist += self.distance_matrix[tour[-1]][tour[0]]
        return total_dist

    def random_solution(self):
        """
        Create a random permutation of city indices.
        """
        tour = list(range(self.num_cities))
        random.shuffle(tour)
        return tour

