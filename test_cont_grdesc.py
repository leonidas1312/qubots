from rastion_core.base_problem import BaseProblem
from rastion_core.algorithms.gradient_descent import GradientDescent
import numpy as np

class QuadraticProblem(BaseProblem):
    def __init__(self):
        # For example, f(x, y) = x^2 + y^2
        self.dimension = 2  # x, y

    def evaluate_solution(self, solution) -> float:
        x, y = solution
        return x**2 + y**2

    def gradient(self, solution):
        x, y = solution
        # Gradient = (2x, 2y)
        return np.array([2*x, 2*y], dtype=float)

    def random_solution(self):
        # random x, y in [-10, 10]
        return np.random.uniform(-10, 10, size=self.dimension)

problem = QuadraticProblem()
gd_solver = GradientDescent(lr=0.1, max_iters=50, verbose=True)
best_sol, best_val = gd_solver.optimize(problem)
print("Best solution found:", best_sol)
print("Best objective value:", best_val)
