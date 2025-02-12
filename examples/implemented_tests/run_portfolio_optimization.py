from qubots.auto_problem import AutoProblem
from qubots.auto_optimizer import AutoOptimizer

def run_portfolio_optimization():
    org = "Rastion"
    
    # Load the portfolio optimization problem.
    # This assumes your problem repository "portfolio-optimization" contains a file
    # named "portfolio_optimization.py" and a valid "problem_config.json".
    problem = AutoProblem.from_repo(f"{org}/portfolio-optimization", revision="main")
    
    # Load the Particle Swarm Optimizer.
    # This assumes your optimizer repository "particle-swarm" contains a file
    # named "particle_swarm_optimizer.py" and a valid "solver_config.json".
    optimizer = AutoOptimizer.from_repo(f"{org}/particle-swarm",
                                        revision="main",
                                        override_params={"swarm_size":60,"max_iters":300})
    
    best_solution, best_cost = optimizer.optimize(problem)
    print("Portfolio Optimization with PSO")
    print("Best Solution:", best_solution)
    print("Best Cost:", best_cost)

if __name__ == "__main__":
    run_portfolio_optimization()
