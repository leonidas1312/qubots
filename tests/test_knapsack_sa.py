# test_knapsack_sa.py
from rastion_core.problems.knapsack import KnapsackProblem
from rastion_core.algorithms.simulated_annealing import SimulatedAnnealing

# sample items
items = [
    (10, 5), # value=10, weight=5
    (6, 3),
    (5, 2),
    (20, 10),
    (2, 1),
]
capacity = 10
problem = KnapsackProblem(items, capacity)

sa_solver = SimulatedAnnealing(
    initial_temp=100,
    cooling_rate=0.95,
    max_iters=200,
    verbose=True
)

best_sol, best_cost = sa_solver.optimize(problem)
print("Knapsack best solution:", best_sol)
print("Knapsack best cost (remember it's negative of total value if feasible):", best_cost)
