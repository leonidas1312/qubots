# test_qubo_sa.py
from rastion_core.problems.qubo_problem import QUBOProblem
from rastion_core.algorithms.simulated_annealing import SimulatedAnnealing

# small Q
Q = {
   (0, 0): 1,   # x0^2
   (1, 1): 2,   # x1^2
   (0, 1): -2,  # x0*x1
}
# dimension=2 => x in {0,1}^2

problem = QUBOProblem(Q)

sa_solver = SimulatedAnnealing(initial_temp=10, cooling_rate=0.9, max_iters=50, verbose=True)
best_sol, best_val = sa_solver.optimize(problem)
print("QUBO best solution:", best_sol)
print("QUBO cost:", best_val)
