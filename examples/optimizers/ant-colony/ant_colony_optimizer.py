from rastion_hub.base_optimizer import BaseOptimizer
import random
import numpy as np

class AntColonyOptimizer(BaseOptimizer):
    """
    A simple Ant Colony Optimization (ACO) implementation
    for combinatorial problems such as the Traveling Salesman Problem.
    """
    def __init__(self, num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5, verbose=False):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.verbose = verbose

    def optimize(self, problem, **kwargs):
        distance_matrix = problem.distance_matrix
        n = len(distance_matrix)
        pheromone = np.ones((n, n))
        best_route = None
        best_distance = float("inf")
        for iteration in range(self.num_iterations):
            all_routes = []
            all_distances = []
            for _ in range(self.num_ants):
                route = [0]
                unvisited = list(range(1, n))
                while unvisited:
                    current = route[-1]
                    probs = []
                    for j in unvisited:
                        tau = pheromone[current][j] ** self.alpha
                        eta = (1.0 / distance_matrix[current][j]) ** self.beta
                        probs.append(tau * eta)
                    sum_probs = sum(probs)
                    probs = [p / sum_probs for p in probs]
                    next_node = random.choices(unvisited, weights=probs, k=1)[0]
                    route.append(next_node)
                    unvisited.remove(next_node)
                route.append(0)
                distance = problem.evaluate_solution(route)
                all_routes.append(route)
                all_distances.append(distance)
                if distance < best_distance:
                    best_distance = distance
                    best_route = route
            pheromone *= (1 - self.evaporation_rate)
            for route, distance in zip(all_routes, all_distances):
                for i in range(len(route)-1):
                    pheromone[route[i]][route[i+1]] += 1.0 / distance
            if self.verbose:
                print(f"ACO Iteration {iteration}: Best Distance = {best_distance}")
        return best_route, best_distance
