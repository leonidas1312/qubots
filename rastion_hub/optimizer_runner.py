"""
optimizer_runner.py

This module provides helper functions to run multiple optimizers on the same problem.

Two common patterns are supported:
1. Independent Runs:
   Run each optimizer separately on the same problem and then compare their results.

2. Chained Refinement:
   Run optimizers sequentially where each optimizer receives the previous solution as its
   initial solution, potentially refining it further.
"""

def run_optimizers_independently(problem, optimizers):
    """
    Run each optimizer independently on the given problem.

    Parameters:
        problem: An instance of a problem (e.g., created via AutoProblem).
        optimizers: A list of optimizer instances (e.g., created via AutoOptimizer).

    Returns:
        A list of tuples: (optimizer_name, solution, cost) for each optimizer.
    """
    results = []
    for optimizer in optimizers:
        optimizer_name = optimizer.__class__.__name__
        print(f"Running {optimizer_name} independently...")
        # Each optimizer starts fresh; no initial_solution is provided.
        solution, cost = optimizer.optimize(problem)
        results.append((optimizer_name, solution, cost))
        print(f"{optimizer_name} produced cost: {cost}\n")
    return results

def run_optimizers_in_chain(problem, optimizers):
    """
    Run a sequence of optimizers on the problem so that each one refines the result
    of the previous optimizer.

    Each optimizer is called with an `initial_solution` parameter (if available),
    allowing the solution to be progressively refined.

    Parameters:
        problem: An instance of a problem (e.g., created via AutoProblem).
        optimizers: A list of optimizer instances (e.g., created via AutoOptimizer).

    Returns:
        A tuple (final_solution, final_cost) after all optimizers have been applied.
    """
    best_solution = None
    best_cost = float("inf")
    
    for optimizer in optimizers:
        optimizer_name = optimizer.__class__.__name__
        print(f"Running {optimizer_name} with initial solution: {best_solution}")
        # Pass the previous solution (if any) as initial_solution.
        best_solution, best_cost = optimizer.optimize(problem, initial_solution=best_solution)
        print(f"{optimizer_name} refined the solution to cost: {best_cost}")
    
    return best_solution, best_cost
