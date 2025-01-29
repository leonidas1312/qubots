from rastion_core.problems.traveling_salesman import TSPProblem
from rastion_core.algorithms.genetic_algorithm import GeneticAlgorithm

# Simple 5-city distance matrix
dist_matrix = [
    [0, 2, 9, 10, 7],
    [2, 0, 6, 4, 5],
    [9, 6, 0, 8, 7],
    [10, 4, 8, 0, 6],
    [7, 5, 7, 6, 0]
]

tsp = TSPProblem(dist_matrix)
solver = GeneticAlgorithm(population_size=50, max_generations=100, verbose=True)
best_tour, best_cost = solver.optimize(tsp)
print("Best TSP tour:", best_tour)
print("Cost:", best_cost)
