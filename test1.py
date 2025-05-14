from qubots.auto_problem import AutoProblem
from qubots.auto_optimizer import AutoOptimizer

problem = AutoProblem.from_repo("ileo/tspqubot1",
                                override_params={"instance_file":"att48.tsp"})

num_of_cities = problem.nb_cities
dist_matrix = problem.dist_matrix

optimizer = AutoOptimizer.from_repo("ileo/ChristofidesTSP_optimizer",
                                    override_params={'time_limit': 10,
                                                     'num_cities': num_of_cities,
                                                     'distance_matrix': dist_matrix})
solution = optimizer.optimize(problem)
print(solution)