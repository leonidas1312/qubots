from qubots.auto_problem import AutoProblem
from qubots.auto_optimizer import AutoOptimizer

problem = AutoProblem.from_repo(f"Rastion/portfolio-optimization", revision="main")
optimizer = AutoOptimizer.from_repo(f"Rastion/particle-swarm",
                                    revision="main",
                                    override_params={"swarm_size":60,"max_iters":500})

best_solution, best_cost = optimizer.optimize(problem)
print("Portfolio Optimization with PSO")
print("Best Solution:", best_solution)
print("Best Cost:", best_cost)