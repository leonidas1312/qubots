from qubots.auto_problem import AutoProblem
from qubots.auto_optimizer import AutoOptimizer

problem = AutoProblem.from_repo(f"Rastion/portfolio-optimization-with-yf", revision="main", override_params={"tickers":['AAPL', 'MSFT', 'GOOG']})
optimizer = AutoOptimizer.from_repo(f"Rastion/quantum-qaoa",
                                    revision="main",
                                    override_params={"num_layers":2,"max_iters":100})

best_solution, best_cost = optimizer.optimize(problem)
print("Portfolio Optimization with QAOA")
print("Best Solution:", best_solution)
print("Best Cost:", best_cost)