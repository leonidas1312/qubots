from qubots.auto_problem import AutoProblem
from qubots.auto_optimizer import AutoOptimizer
from qubots.optimizer_runner import run_optimizers_independently, run_optimizers_in_chain

def run_maxcut_optimization_independently():
    """
    Load the maxcut optimization problem and run multiple optimizers
    independently (each with no initial solution). The results are then compared.
    """
    org = "Rastion"
    
    # Load the maxcut optimization problem.
    problem = AutoProblem.from_repo(f"{org}/max-cut", revision="main")
    
    # Load several optimizers with optional parameter overrides.
    optimizer1 = AutoOptimizer.from_repo(
        f"{org}/particle-swarm",
        revision="main",
        override_params={"swarm_size": 50, "max_iters": 20}
    )
    optimizer2 = AutoOptimizer.from_repo(
        f"{org}/tabu-search",
        revision="main",
        override_params={"tabu_tenure": 10, "max_iters": 20}
    )
    optimizer3 = AutoOptimizer.from_repo(
        f"{org}/rl-optimizer",
        revision="main",
        override_params={"time_limit": 1}#1 seconds
    )
    
    optimizers = [optimizer1, optimizer2, optimizer3]
    
    results = run_optimizers_independently(problem, optimizers)
    
    # Find the best result (assuming lower cost is better).
    best_optimizer, best_solution, best_cost = min(results, key=lambda x: x[2])
    
    print("=== Independent Runs Results ===")
    for name, sol, cost in results:
        print(f"Optimizer {name}: Cost = {cost}, Solution = {sol}")
    print(f"\nBest optimizer: {best_optimizer} with cost = {best_cost}, solution = {best_solution}\n")


def run_maxcut_optimization_chained():
    """
    Load the maxcut optimization problem and run a chain of optimizers sequentially.
    Each optimizer refines the solution provided by its predecessor.
    """
    org = "Rastion"
    
    # Load the maxcut optimization problem.
    problem = AutoProblem.from_repo(f"{org}/max-cut", revision="main")
    
    # Load a chain of optimizers.
    optimizer1 = AutoOptimizer.from_repo(
        f"{org}/particle-swarm",
        revision="main",
        override_params={"swarm_size": 50, "max_iters": 20}
    )
    optimizer2 = AutoOptimizer.from_repo(
        f"{org}/tabu-search",
        revision="main",
        override_params={"tabu_tenure": 10, "max_iters": 20}
    )
    optimizer3 = AutoOptimizer.from_repo(
        f"{org}/rl-optimizer",
        revision="main",
        override_params={"time_limit": 1}#1 seconds
    )
    
    optimizers_chain = [optimizer1, optimizer2, optimizer3]
    
    final_solution, final_cost = run_optimizers_in_chain(problem, optimizers_chain)
    
    print("=== Chained Refinement Results ===")
    print(f"Final refined solution: {final_solution} with cost: {final_cost}\n")


def main():
    print("=== Running Maxcut Optimization with Independent Optimizer Runs ===")
    run_maxcut_optimization_independently()
    
    print("=== Running Maxcut Optimization with Chained Refinement ===")
    run_maxcut_optimization_chained()


if __name__ == "__main__":
    main()
