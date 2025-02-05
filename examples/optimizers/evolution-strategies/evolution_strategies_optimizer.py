from rastion_hub.base_optimizer import BaseOptimizer
import numpy as np
import random

class EvolutionStrategiesOptimizer(BaseOptimizer):
    """
    A simple Evolution Strategies (ES) optimizer for continuous problems.
    """
    def __init__(self, population_size=50, max_iters=100, sigma=0.1, learning_rate=0.01, verbose=False):
        self.population_size = population_size
        self.max_iters = max_iters
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.verbose = verbose

    def optimize(self, problem, initial_solution=None, **kwargs):
        if initial_solution is None:
            x = np.array(problem.random_solution(), dtype=float)
        else:
            x = np.array(initial_solution, dtype=float)
        best_solution = x.copy()
        best_score = problem.evaluate_solution(x)
        for iter in range(self.max_iters):
            gradients = np.zeros_like(x)
            for _ in range(self.population_size):
                noise = np.random.randn(*x.shape) * self.sigma
                candidate = x + noise
                score = problem.evaluate_solution(candidate)
                gradients += noise * score
                if score < best_score:
                    best_score = score
                    best_solution = candidate.copy()
            x = x - self.learning_rate * gradients / self.population_size
            if self.verbose:
                print(f"ES Iteration {iter}: Best Score = {best_score}")
        return best_solution, best_score
