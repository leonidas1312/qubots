# rastion_core/algorithms/hill_climb.py

from copy import deepcopy
import random
from rastion_core.base_optimizer import BaseOptimizer

class HillClimb(BaseOptimizer):
    """
    A basic hill-climbing approach.
    - In each iteration, pick some neighbors, move to best neighbor if better.
    """

    def __init__(self, max_iters=1000, neighbor_func=None, num_neighbors=10, verbose=False):
        self.max_iters = max_iters
        self.neighbor_func = neighbor_func
        self.num_neighbors = num_neighbors
        self.verbose = verbose

    def optimize(self, problem, initial_solution=None, **kwargs):
        if initial_solution is None:
            current_sol = problem.random_solution()
        else:
            current_sol = deepcopy(initial_solution)

        current_cost = problem.evaluate_solution(current_sol)
        best_sol = deepcopy(current_sol)
        best_cost = current_cost

        for i in range(self.max_iters):
            # generate neighbors
            neighbors = [self.neighbor_func(current_sol) for _ in range(self.num_neighbors)]
            best_neighbor = None
            best_neighbor_cost = float("inf")
            for nb in neighbors:
                cost_nb = problem.evaluate_solution(nb)
                if cost_nb < best_neighbor_cost:
                    best_neighbor_cost = cost_nb
                    best_neighbor = nb

            if best_neighbor_cost < current_cost:
                current_sol = best_neighbor
                current_cost = best_neighbor_cost
                if current_cost < best_cost:
                    best_sol, best_cost = current_sol, current_cost
            else:
                # no improvement => end or continue
                break

            if self.verbose:
                print(f"Iteration {i}, cost={current_cost}")

        return best_sol, best_cost
