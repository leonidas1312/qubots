"""
usage/test_continuous_problems.py

This script tests continuous optimization problems and solvers.
We define three continuous problems (Quadratic, Rosenbrock, and Rastrigin)
and an exhaustive grid search optimizer (for small dimensions) that
is guaranteed to find the best (discretized) solution.
Then, we load three continuous solvers from the hub and compare their
results against the grid-search “optimal” solution.
"""

import numpy as np
import math
import itertools
import time

from qubots.base_problem import BaseProblem
from qubots.auto_optimizer import AutoOptimizer

#############################
# Define Test Continuous Problems
#############################

class QuadraticProblem(BaseProblem):
    """
    A simple quadratic function:
         f(x) = sum((x_i - 3)^2)
    Global optimum at x = [3,3,...,3] with cost 0.
    Domain: each x_i in [0,6]
    """
    def __init__(self, dim=3):
        self.dim = dim
        self.lower_bound = [0.0] * dim
        self.upper_bound = [6.0] * dim
        self.optimization_type = "continuous"
    
    def evaluate_solution(self, solution) -> float:
        x = np.array(solution)
        return float(np.sum((x - 3)**2))
    
    def random_solution(self):
        return list(np.random.uniform(0, 6, self.dim))


class RosenbrockProblem(BaseProblem):
    """
    The Rosenbrock function (2D version):
         f(x, y) = 100*(y - x^2)^2 + (1 - x)^2
    Global optimum at (1,1) with cost 0.
    Domain: each variable in [-2,2]
    """
    def __init__(self, dim=2):
        self.dim = dim
        self.lower_bound = [-2.0] * dim
        self.upper_bound = [2.0] * dim
        self.optimization_type = "continuous"
    
    def evaluate_solution(self, solution) -> float:
        x = np.array(solution)
        cost = 0.0
        for i in range(self.dim - 1):
            cost += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return float(cost)
    
    def random_solution(self):
        return list(np.random.uniform(-2, 2, self.dim))


class RastriginProblem(BaseProblem):
    """
    The Rastrigin function (2D version):
         f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    Global optimum at x = [0,0] with cost 0.
    Domain: each x_i in [-5.12, 5.12]
    """
    def __init__(self, dim=2):
        self.dim = dim
        self.lower_bound = [-5.12] * dim
        self.upper_bound = [5.12] * dim
        self.optimization_type = "continuous"
    
    def evaluate_solution(self, solution) -> float:
        x = np.array(solution)
        A = 10
        cost = A * self.dim + np.sum(x**2 - A * np.cos(2 * np.pi * x))
        return float(cost)
    
    def random_solution(self):
        return list(np.random.uniform(-5.12, 5.12, self.dim))


#############################
# Exhaustive (Grid) Search Optimizer for Continuous Problems
#############################

class ExhaustiveSearchContinuous:
    """
    A grid search optimizer for continuous problems.
    Given a problem that defines lower_bound and upper_bound (as lists) and is low-dimensional,
    it discretizes each dimension (with a given resolution) and evaluates all grid points.
    """
    def __init__(self, resolution=20):
        self.resolution = resolution

    def optimize(self, problem, **kwargs):
        dims = len(problem.lower_bound)
        # Create a grid for each dimension.
        grids = [
            np.linspace(lb, ub, self.resolution)
            for lb, ub in zip(problem.lower_bound, problem.upper_bound)
        ]
        best_solution = None
        best_cost = float('inf')
        # Iterate over all grid points.
        for point in itertools.product(*grids):
            point = np.array(point)
            cost = problem.evaluate_solution(point)
            if cost < best_cost:
                best_cost = cost
                best_solution = point
        return best_solution, best_cost


#############################
# Test Runner for a Single Problem
#############################

def test_continuous_problem(problem, problem_name, solvers, grid_solver):
    print("=" * 60)
    print(f"Testing {problem_name} Problem")
    print("=" * 60)
    
    start_time = time.time()
    optimal_solution, optimal_cost = grid_solver.optimize(problem)
    grid_time = time.time() - start_time
    print("Exhaustive (Grid) Search:")
    print("  Optimal Solution:", optimal_solution)
    print("  Optimal Cost:    {:.6f}".format(optimal_cost))
    print("  Time taken:      {:.4f} sec\n".format(grid_time))
    
    # For each solver, load the solver using AutoOptimizer and run the optimization.
    for solver_name, solver_config in solvers.items():
        print("-" * 60)
        print(f"Using Solver: {solver_name}")
        print("-" * 60)
        
        # Load the solver from the hub (assuming the repo names are as provided).
        solver = AutoOptimizer.from_repo(
            f"Rastion/{solver_config['repo']}",
            revision="main",
            override_params=solver_config.get("override_params", {})
        )
        
        # (Optionally) Check compatibility based on the optimization_type.
        if hasattr(solver, "optimization_type"):
            if solver.optimization_type != problem.optimization_type:
                print(f"Skipping {solver_name}: Incompatible optimization type.")
                continue
        
        start = time.time()
        solution, cost = solver.optimize(problem)
        solver_time = time.time() - start
        
        print(f"{solver_name} returned:")
        print("  Solution:", solution)
        print("  Cost:     {:.6f}".format(cost))
        print("  Time:     {:.4f} sec".format(solver_time))
        
        error = abs(cost - optimal_cost)
        rel_error = error / (abs(optimal_cost) if optimal_cost != 0 else 1)
        print("  Absolute Error: {:.6f}".format(error))
        print("  Relative Error: {:.2f}%\n".format(rel_error * 100))


#############################
# Main Function: Run Tests on All Problems
#############################

def main():
    # Define our test problems.
    quadratic = QuadraticProblem(dim=3)
    rosenbrock = RosenbrockProblem(dim=2)
    rastrigin = RastriginProblem(dim=2)
    
    # Instantiate our exhaustive search optimizer (grid search) with a suitable resolution.
    grid_solver = ExhaustiveSearchContinuous(resolution=50)
    
    # Define the continuous solvers to test.
    # These repo names should match the ones used in your hub.
    solvers = {
        "Differential Evolution": {
            "repo": "differential-evolution",
            "override_params": {}  # You can override parameters if desired.
        },
        "Evolution Strategies": {
            "repo": "evolution-strategies",
            "override_params": {}
        },
    }
    
    # Test each problem.
    test_continuous_problem(quadratic, "Quadratic", solvers, grid_solver)
    test_continuous_problem(rosenbrock, "Rosenbrock", solvers, grid_solver)
    test_continuous_problem(rastrigin, "Rastrigin", solvers, grid_solver)


if __name__ == "__main__":
    main()
