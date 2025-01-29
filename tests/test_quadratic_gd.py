# test_quadratic_gd.py
import numpy as np
from rastion_core.problems.quadratic import QuadraticProblem
from rastion_core.algorithms.gradient_descent import GradientDescent

# Q is 2x2
Q = np.array([[2, 0], [0, 5]])
problem = QuadraticProblem(dimension=2, Q=Q)

gd = GradientDescent(lr=0.05, max_iters=100, verbose=True)
sol, val = gd.optimize(problem)
print("Quadratic best sol:", sol)
print("Quadratic best val:", val)
