from rastion_core.base_optimizer import BaseOptimizer
import random
import copy

class TabuSearchOptimizer(BaseOptimizer):
    """
    A simple Tabu Search implementation.
    Assumes that problem.random_solution() returns a solution (e.g., a list)
    and problem.evaluate_solution(solution) returns a cost to minimize.
    """
    def __init__(self, max_iters=100, tabu_tenure=5, verbose=False):
        self.max_iters = max_iters
        self.tabu_tenure = tabu_tenure
        self.verbose = verbose

    def optimize(self, problem, **kwargs):
        current_solution = problem.random_solution()
        best_solution = copy.deepcopy(current_solution)
        best_score = problem.evaluate_solution(best_solution)
        tabu_list = []
        for iter in range(self.max_iters):
            neighbors = []
            # Generate neighbors by swapping two random indices (if list)
            if isinstance(current_solution, list) and len(current_solution) >= 2:
                for _ in range(10):
                    neighbor = current_solution.copy()
                    i, j = random.sample(range(len(neighbor)), 2)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    neighbors.append(neighbor)
            else:
                neighbors.append(problem.random_solution())
            feasible_neighbors = [n for n in neighbors if n not in tabu_list]
            if not feasible_neighbors:
                feasible_neighbors = neighbors
            candidate = min(feasible_neighbors, key=lambda n: problem.evaluate_solution(n))
            candidate_score = problem.evaluate_solution(candidate)
            if candidate_score < best_score:
                best_solution = candidate
                best_score = candidate_score
            tabu_list.append(candidate)
            if len(tabu_list) > self.tabu_tenure:
                tabu_list.pop(0)
            current_solution = candidate
            if self.verbose:
                print(f"Tabu Iteration {iter}: Best Score = {best_score}")
        return best_solution, best_score
