from rastion_core.base_optimizer import BaseOptimizer
import random
import numpy as np

class DifferentialEvolution(BaseOptimizer):
    """
    Differential Evolution (DE) solver for continuous optimization.
    """
    def __init__(self, population_size=50, max_iters=100, F=0.8, CR=0.9, verbose=False):
        self.population_size = population_size
        self.max_iters = max_iters
        self.F = F
        self.CR = CR
        self.verbose = verbose

    def optimize(self, problem, bounds=None, **kwargs):
        if bounds is None:
            x0 = np.array(problem.random_solution(), dtype=float)
            dim = len(x0)
            bounds = [(-10, 10)] * dim
        else:
            dim = len(bounds)
        population = []
        for _ in range(self.population_size):
            ind = np.array([random.uniform(b[0], b[1]) for b in bounds])
            population.append(ind)
        best_ind = min(population, key=lambda ind: problem.evaluate_solution(ind))
        best_score = problem.evaluate_solution(best_ind)
        for iter in range(self.max_iters):
            new_population = []
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = random.sample(indices, 3)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, [lb for lb, ub in bounds], [ub for lb, ub in bounds])
                cross_points = np.random.rand(dim) < self.CR
                if not any(cross_points):
                    cross_points[random.randint(0, dim-1)] = True
                trial = np.where(cross_points, mutant, population[i])
                if problem.evaluate_solution(trial) < problem.evaluate_solution(population[i]):
                    new_population.append(trial)
                else:
                    new_population.append(population[i])
            population = new_population
            current_best = min(population, key=lambda ind: problem.evaluate_solution(ind))
            current_best_score = problem.evaluate_solution(current_best)
            if current_best_score < best_score:
                best_score = current_best_score
                best_ind = current_best
            if self.verbose:
                print(f"DE Iteration {iter}: Best Score = {best_score}")
        return best_ind, best_score
