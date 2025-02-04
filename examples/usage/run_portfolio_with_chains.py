from rastion_hub.auto_problem import AutoProblem
from rastion_hub.auto_optimizer import AutoOptimizer
from rastion_hub.optimizer_runner import run_optimizers_independently, run_optimizers_in_chain

def run_portfolio_optimization_independently():
    """
    Load the portfolio optimization problem and run multiple optimizers
    independently (each with no initial solution). The results are then compared.
    """
    org = "Rastion"
    
    # Load the portfolio optimization problem.
    # This assumes your "portfolio-optimization" repo contains the necessary files.
    problem = AutoProblem.from_repo(f"{org}/portfolio-optimization", revision="main")
    
    # Load several optimizers with optional parameter overrides.
    optimizer1 = AutoOptimizer.from_repo(
        f"{org}/particle-swarm",
        revision="main",
        override_params={"swarm_size": 60, "max_iters": 100}
    )
    optimizer2 = AutoOptimizer.from_repo(
        f"{org}/differential-evolution",
        revision="main",
        override_params={"population_size": 50, "max_iters": 100}
    )
    optimizer3 = AutoOptimizer.from_repo(
        f"{org}/evolution-strategies",
        revision="main",
        override_params={"population_size": 50, "max_iters": 100}
    )
    
    optimizers = [optimizer1, optimizer2, optimizer3]
    
    results = run_optimizers_independently(problem, optimizers)
    
    # Find the best result (assuming lower cost is better).
    best_optimizer, best_solution, best_cost = min(results, key=lambda x: x[2])
    
    print("=== Independent Runs Results ===")
    for name, sol, cost in results:
        print(f"Optimizer {name}: Cost = {cost}, Solution = {sol}")
    print(f"\nBest optimizer: {best_optimizer} with cost = {best_cost}, solution = {best_solution}\n")


def run_portfolio_optimization_chained():
    """
    Load the portfolio optimization problem and run a chain of optimizers sequentially.
    Each optimizer refines the solution provided by its predecessor.
    """
    org = "Rastion"
    
    # Load the portfolio optimization problem.
    problem = AutoProblem.from_repo(f"{org}/portfolio-optimization", revision="main")
    
    # Load a chain of optimizers.
    # For example, start with a global search (Particle Swarm), then refine using
    # Differential Evolution and finally Evolution Strategies.
    optimizer1 = AutoOptimizer.from_repo(
        f"{org}/particle-swarm",
        revision="main",
        override_params={"swarm_size": 60, "max_iters": 100}
    )
    optimizer2 = AutoOptimizer.from_repo(
        f"{org}/differential-evolution",
        revision="main",
        override_params={"population_size": 50, "max_iters": 100}
    )
    optimizer3 = AutoOptimizer.from_repo(
        f"{org}/evolution-strategies",
        revision="main",
        override_params={"population_size": 50, "max_iters": 100}
    )
    
    optimizers_chain = [optimizer1, optimizer2, optimizer3]
    
    final_solution, final_cost = run_optimizers_in_chain(problem, optimizers_chain)
    
    print("=== Chained Refinement Results ===")
    print(f"Final refined solution: {final_solution} with cost: {final_cost}\n")


def main():
    print("=== Running Portfolio Optimization with Independent Optimizer Runs ===")
    run_portfolio_optimization_independently()
    
    print("=== Running Portfolio Optimization with Chained Refinement ===")
    run_portfolio_optimization_chained()


if __name__ == "__main__":
    main()
