from qubots.auto_optimizer import AutoOptimizer
from qubots.auto_problem import AutoProblem



problem = AutoProblem.from_repo("Rastion/max-cut", revision="main", override_params={"num_nodes":6})


# QAOA for QUBO:
qaoa_optimizer = AutoOptimizer.from_repo(
    "Rastion/quantum-qaoa",   # your GitHub repo name for the QAOA optimizer
    revision="main",
    override_params={"num_layers": 2, "max_iters": 100, "verbose": True}  # if you wish to override defaults
)

solution, cost = qaoa_optimizer.optimize(problem)
print("QAOA solution:", solution, "with cost:", cost)

# VQE for QUBO:
vqe_optimizer = AutoOptimizer.from_repo(
    "Rastion/quantum-vqe",
    revision="main",
    override_params={"num_layers": 4, "max_iters": 100, "verbose": True}  # if you wish to override defaults
)
solution, cost = vqe_optimizer.optimize(problem)
print("VQE solution:", solution, "with cost:", cost)

# Exhaustive search for small problems
exh_opt =AutoOptimizer.from_repo("Rastion/exhaustive-search")

solution, cost = exh_opt.optimize(problem)
print("Optimal solution:", solution, "with cost:", cost)
