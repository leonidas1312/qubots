from rastion_hub.auto_problem import AutoProblem
from rastion_hub.auto_optimizer import AutoOptimizer
from rastion_hub.optimizer_runner import run_optimizers_in_chain
# Load a small maxcut optimization problem.
problem = AutoProblem.from_repo(f"Rastion/max-cut", revision="main", override_params={"num_nodes": 8})

# Load a chain of optimizers.
# For example, start with a global search particle swarm, then refine using
# tabu search and finally the custom rl optimizer.
optimizer1 = AutoOptimizer.from_repo(
   f"Rastion/particle-swarm",
   revision="main",
   override_params={"swarm_size": 50, "max_iters": 100}
)
optimizer2 = AutoOptimizer.from_repo(
   f"Rastion/tabu-search",
   revision="main",
   override_params={"max_iters": 100, "tabu_tenure": 10, "verbose": True}
)
optimizer3 = AutoOptimizer.from_repo(
   f"Rastion/rl-optimizer",
   revision="main",
   override_params={"time_limit": 1  # seconds
   
   }
)

optimizers_chain = [optimizer1, optimizer2, optimizer3]

final_solution, final_cost = run_optimizers_in_chain(problem, optimizers_chain)

print("=== Chained Refinement Results ===")
print(f"Final refined solution: {final_solution} with cost: {final_cost}\n")

exhaustive_optimizer = AutoOptimizer.from_repo(
   f"Rastion/exhaustive-search",
   revision="main",
)

best_solution, best_cost = exhaustive_optimizer.optimize(problem)
print("=== Exhaustive Results ===")
print(f"Best solution: {best_solution} with cost: {best_cost}\n")