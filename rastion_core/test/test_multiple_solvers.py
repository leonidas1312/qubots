from rastion_core.base_optimizer import BaseOptimizer
from rastion_core.problems.traveling_salesman import TSPProblem
from rastion_core.algorithms.genetic_algorithm import GeneticAlgorithm
from rastion_core.algorithms.gradient_descent import GradientDescent

# If you had multiple solvers, you might store them in a list:
solvers = [
    GeneticAlgorithm(population_size=50, max_generations=100, verbose=True),
    GradientDescent(lr=0.01, max_iters=100)
]

# Simple 5-city distance matrix
dist_matrix = [
    [0, 2, 9, 10, 7],
    [2, 0, 6, 4, 5],
    [9, 6, 0, 8, 7],
    [10, 4, 8, 0, 6],
    [7, 5, 7, 6, 0]
]

problem = TSPProblem(dist_matrix)

for solver in solvers:
    assert isinstance(solver, BaseOptimizer)  # check for uniform interface
    best_sol, best_cost = solver.optimize(problem)
    print(solver, best_sol, best_cost)
