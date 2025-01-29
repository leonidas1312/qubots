# tests/test_git_repo.py

import pytest
from rastion_hub.auto_optimizer import AutoOptimizer
from rastion_core.problems.traveling_salesman import TSPProblem

def test_clone_solver_tsp():
    """
    Suppose we have a GitHub repo at: github.com/rastion-hub/genetic_tsp_v1
    containing solver_config.json -> "entry_point": "my_ga_module.solver:GeneticTSPSolver"
    """
    # We'll skip if no network
    # or we can do a check if 'git' is installed

    solver = AutoOptimizer.from_repo("rastion-hub/genetic_tsp_v1", revision="main")
    # Then create a TSP problem
    dist_matrix = [
        [0, 2, 9],
        [2, 0, 6],
        [9, 6, 0]
    ]
    problem = TSPProblem(dist_matrix)
    best_sol, best_cost = solver.optimize(problem)
    assert best_sol is not None
    print("TSP solver with GA from GitHub -> best_sol:", best_sol, " best_cost:", best_cost)
